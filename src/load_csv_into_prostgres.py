import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

# Load CSV
df = pd.read_csv("data/Software_Salaries.csv")

# Raw password
raw_password = "1{Rithwik}"

# URL-encode it
safe_password = urllib.parse.quote_plus(raw_password)

# Create SQLAlchemy engine
engine = create_engine(f'postgresql+psycopg2://postgres:{safe_password}@localhost:5432/postgres')

# Write to PostgreSQL (replace if exists)
df.to_sql('salary_data', engine, if_exists='replace', index=False)

print("Data loaded into PostgreSQL successfully!")