from data_loader import load_data_from_postgres
from data_cleaner import clean_data
from data_preprocessor import preprocess_data
from model_training_evaluation_selector import train_evaluate_and_select_model

def run_pipeline():
    """
    Full pipeline to run data ingestion, cleaning, preprocessing, and training.
    """
    print(" Starting pipeline...")

    # Step 1: Load data
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

    # Step 2: Clean data
    df_cleaned = clean_data(df)
    print(" Data cleaned.")

    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test, y_transformer = preprocess_data(df_cleaned)

    # Step 4: Train model & log with MLflow
    best_model, y_pred_test = train_evaluate_and_select_model(X_train, y_train, X_test, y_test, save_dir="pkl_joblib_files")
    print(" Model trained evaluated selected and logged.")

    print(" Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()