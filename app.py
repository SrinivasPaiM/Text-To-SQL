
import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load AI model for SQL generation
MODEL_NAME = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

def generate_sql(nl_query, schema):
    """Generate SQL query from a natural language query using AI."""
    prompt = f"""### Database Schema:
{schema}

### Convert the following question into an SQL query:
{nl_query}

SQL Query:
"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device, dtype=torch.long)  # Ensure Long dtype

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)

    # If model outputs in float16 or bfloat16, convert back to long/int
    if output_ids.dtype in [torch.float16, torch.bfloat16]:
        output_ids = output_ids.to(dtype=torch.long)

    # Decode and clean the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    sql_start = output_text.find("SQL Query:") + len("SQL Query:")
    sql_query = output_text[sql_start:].strip()

    # Clean SQL output
    sql_query = re.sub(r"```sql|```", "", sql_query).split("###")[0].strip()
    return sql_query


def execute_sql(sql_query, db_path):
    """Execute the generated SQL query on the provided database."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        return df
    except Exception as e:
        return str(e)

def get_schema(db_path):
    """Extract schema from the uploaded database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = ""

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema += f"TABLE {table_name} (\n"
        schema += ",\n".join([f"  {col[1]} {col[2]}" for col in columns])
        schema += "\n);\n\n"

    conn.close()
    return schema

# --- Streamlit UI ---
st.title("AI-Powered Text-to-SQL Generator")
st.write("Convert natural language questions into SQL queries and execute them.")

# Database selection
db_option = st.radio("How do you want to provide your database?", ["Upload .db file", "Enter schema manually"])

db_path = None
schema = ""

if db_option == "Upload .db file":
    uploaded_file = st.file_uploader("Upload a SQLite `.db` file", type=["db"])
    if uploaded_file:
        db_path = "uploaded_database.db"
        with open(db_path, "wb") as f:
            f.write(uploaded_file.read())
        schema = get_schema(db_path)
        st.success("✅ Database uploaded successfully!")

elif db_option == "Enter schema manually":
    st.write("Example schema format:")
    st.code(
        """TABLE employees (
  employee_id INT PRIMARY KEY,
  first_name TEXT,
  last_name TEXT,
  salary INT
);""",
        language="sql",
    )
    schema = st.text_area("Enter your schema:")

if schema:
    st.subheader("Extracted/Provided Schema:")
    st.code(schema, language="sql")

# Query input
user_query = st.text_area("📝 Enter your natural language query:")

if st.button("Generate SQL Query"):
    if not schema:
        st.error("❌ Please provide a database or schema first.")
    else:
        sql_query = generate_sql(user_query, schema)
        st.subheader("Generated SQL Query:")
        st.code(sql_query, language="sql")

        # Execute SQL if database exists
        if db_path:
            result = execute_sql(sql_query, db_path)

            if isinstance(result, pd.DataFrame):
                st.subheader("📊 Query Results:")
                st.dataframe(result)

                # Visualization
                if not result.empty:
                    st.subheader("📈 Data Visualization")
                    fig = px.bar(result, x=result.columns[0], y=result.columns[1])
                    st.plotly_chart(fig)
            else:
                st.error(f"❌ SQL Execution Error: {result}")
        else:
            st.info("No database provided, only SQL query was generated.")
