import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

def load_data_from_postgres(
    table_name,
    db_user='postgres',
    db_password='your_password',
    db_host='localhost',
    db_port='5432',
    db_name='your_db'
):
    """
    Load the dataset from a PostgreSQL table and validate its structure.

    Args:
        table_name (str): Name of the table in PostgreSQL.
        db_user (str): Database username.
        db_password (str): Database password.
        db_host (str): Hostname of the database server.
        db_port (str): Port number of the database server.
        db_name (str): Name of the PostgreSQL database.

    Returns:
        pandas.DataFrame: Loaded dataset.

    Raises:
        ValueError: If required columns are missing or the dataset is empty.
    """

    # Encode password to be URL-safe
    encoded_password = urllib.parse.quote_plus(db_password)

    # Create database connection string
    connection_str = f'postgresql+psycopg2://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_str)

    # Load data from PostgreSQL table
    df = pd.read_sql_table(table_name, con=engine)

    # Check if dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty")

    # Define required columns
    required_columns = [
        'job_title', 'experience_level', 'employment_type', 'company_size',
        'company_location', 'remote_ratio', 'salary_currency', 'years_experience', 'base_salary',
        'bonus', 'stock_options', 'total_salary', 'salary_in_usd', 'currency',
        'conversion_rate', 'adjusted_total_usd'
    ]

    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return df


if __name__ == "__main__":
    try:
        df = load_data_from_postgres(
            table_name='salary_data',
            db_user='postgres',
            db_password='1{Rithwik}',
            db_host='localhost',
            db_port='5432',
            db_name='postgres'
        )
        print("Dataset loaded successfully with shape:", df.shape)
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")