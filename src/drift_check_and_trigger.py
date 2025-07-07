import pandas as pd
import os
import mlflow
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset #, TargetDriftPreset

# def generate_all_drift_reports_with_mlflow(
#     train_path="data/train.csv",
#     test_path="data/test.csv",
#     new_path="data/new_data.csv",
#     target_column="adjusted_total_usd",
#     output_dir="drift_reports",
#     experiment_name="salary_prediction_experiment",
#     run_name=None
# ):
#     """
#     Generate Evidently drift reports and log drift metrics and HTML reports to MLflow.
    
#     Returns:
#         dict: Drift scores for train vs test and train vs new data
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # Load data
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)
#     new_df = pd.read_csv(new_path)

#     if target_column not in train_df.columns:
#         raise ValueError(f"Target column '{target_column}' not found in train data.")
    
#     # Split features and target
#     X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]
#     X_test, y_test = test_df.drop(columns=[target_column]), test_df[target_column]
#     X_new, y_new = new_df.drop(columns=[target_column]), new_df[target_column]

#     results = {}

#     mlflow.set_tracking_uri("http://localhost:5000")
#     mlflow.set_experiment(experiment_name)

#     with mlflow.start_run(run_name=run_name or f"drift_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

#         def _run_and_log(ref_X, cur_X, ref_y, cur_y, label):
#             scores = {}

#             # Data Drift
#             dr_report = Report(metrics=[DataDriftPreset()])
#             dr_report.run(reference_data=ref_X, current_data=cur_X)
#             dr_score = dr_report.as_dict()["metrics"][0]["result"]["dataset_drift"]
#             dr_path = os.path.join(output_dir, f"data_drift_{label}.html")
#             dr_report.save_html(dr_path)
#             mlflow.log_metric(f"{label}_data_drift", dr_score)
#             mlflow.log_artifact(dr_path, artifact_path="drift_reports")
#             scores["data_drift"] = dr_score

#             # Target Drift
#             td_report = Report(metrics=[TargetDriftPreset()])
#             td_report.run(
#                 reference_data=pd.DataFrame({target_column: ref_y}),
#                 current_data=pd.DataFrame({target_column: cur_y})
#             )
#             td_score = td_report.as_dict()["metrics"][0]["result"]["dataset_drift"]
#             td_path = os.path.join(output_dir, f"target_drift_{label}.html")
#             td_report.save_html(td_path)
#             mlflow.log_metric(f"{label}_target_drift", td_score)
#             mlflow.log_artifact(td_path, artifact_path="drift_reports")
#             scores["target_drift"] = td_score

#             return scores

#         # Run comparisons
#         results["train_test"] = _run_and_log(X_train, X_test, y_train, y_test, "train_test")
#         results["train_new"] = _run_and_log(X_train, X_new, y_train, y_new, "train_new")

#         print("\nDrift Scores:")
#         for key, val in results.items():
#             print(f"{key}: data_drift={val['data_drift']:.3f}, target_drift={val['target_drift']:.3f}")

#         print(f"\nMLflow Run ID: {run.info.run_id}")
#         print(f"Reports saved in: {output_dir}/")

#     return results

from sklearn.model_selection import train_test_split

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

def split_data(df, target_column="adjusted_total_usd"):
    """
    Splits the input DataFrame into train, validation, and test sets (X and y).
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split: Train vs Temp (Val+Test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Second split: Val vs Test (from temp)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

import json
from pathlib import Path
from datetime import datetime

import mlflow
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

def log_evidently_report(reference_data, current_data, dataset_name="train_vs_test"):
    
    #  Align columns: use only the intersection to avoid partial-column errors
    common_cols = set(reference_data.columns).intersection(current_data.columns)
    if not common_cols:
        print(f"⚠️ No common columns between reference and {dataset_name}; skipping Evidently report.")
        return
    ref = reference_data[sorted(common_cols)]
    cur = current_data[sorted(common_cols)]

    #  Run the Evidently report (drift + summary)
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(reference_data=ref, current_data=cur)

    #  Ensure local save directory exists
    save_dir = Path.cwd() / "evidently_reports"
    save_dir.mkdir(parents=True, exist_ok=True)

    #  Save HTML and JSON
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    html_path = save_dir / f"evidently_{dataset_name}_{ts}.html"
    json_path = save_dir / f"evidently_{dataset_name}_{ts}.json"
    result.save_html(str(html_path))
    with open(json_path, "w", encoding="utf-8") as fp:
        fp.write(result.json())

    #  Log artifacts to MLflow
    mlflow.log_artifact(str(html_path), artifact_path="evidently")
    mlflow.log_artifact(str(json_path), artifact_path="evidently")
    print(f"📄 Logged HTML: {html_path.name}")
    print(f"🗄️  Logged JSON: {json_path.name}")

    #  Load JSON and extract metrics list
    with open(json_path, "r", encoding="utf-8") as fp:
        report_json = json.load(fp)
    metrics_list = report_json.get("metrics", [])

    #  Overall drifted columns metrics
    drift_entry = next((m for m in metrics_list if m.get("metric_id", "").startswith("DriftedColumnsCount")), None)
    if drift_entry:
        count = drift_entry["value"]["count"]
        share = drift_entry["value"]["share"]
        mlflow.log_metric("drifted_columns_count", float(count))
        mlflow.log_metric("drifted_columns_share", float(share))
        print(f"🔢 drifted_columns_count = {count}")
        print(f"🔢 drifted_columns_share = {share}")
    else:
        print("⚠️ No DriftedColumnsCount entry found.")

    #  Row and column counts
    rowcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "RowCount()"), None)
    colcount = next((m["value"] for m in metrics_list if m.get("metric_id") == "ColumnCount()"), None)
    if rowcount is not None:
        mlflow.log_metric("dataset_row_count", float(rowcount))
        print(f"🔢 dataset_row_count = {rowcount}")
    if colcount is not None:
        mlflow.log_metric("dataset_column_count", float(colcount))
        print(f"🔢 dataset_column_count = {colcount}")

    #  Per-feature value drift metrics
    for m in metrics_list:
        mid = m.get("metric_id", "")
        if mid.startswith("ValueDrift(column="):
            # extract column name
            col = mid.split("=")[1].rstrip(")")
            val = m.get("value")
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"drift_{col}", float(val))
                print(f"🔢 drift_{col} = {val}")
    
    print("✅ All requested drift & dataset metrics logged to MLflow.")

import os
import pandas as pd
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "Instilit AI Evidently"



# ---------- Trigger Airflow DAG ----------
def trigger_airflow_dag():
    payload = {"dag_run_id": f"trigger_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    response = requests.post(
        AIRFLOW_API_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        auth=("airflow", "airflow")  # update if different
    )
    if response.status_code == 200:
        print("Airflow DAG triggered successfully.")
    else:
        print("Failed to trigger Airflow DAG:", response.text)

def main():
    client = MlflowClient()

    # ─── 1️⃣ Ensure the MLflow experiment exists and is active ───
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"✅ Created new experiment '{EXPERIMENT_NAME}' (ID={exp_id})")
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        print(f"🔄 Restored deleted experiment '{EXPERIMENT_NAME}' (ID={exp.experiment_id})")
    else:
        print(f"ℹ️ Using existing experiment '{EXPERIMENT_NAME}' (ID={exp.experiment_id})")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ─── 2️⃣ Start your MLflow run ───
    with mlflow.start_run(run_name="Preprocessing and Tuning"):
        # Load and split
        df = load_data_from_postgres(
            table_name='salary_data',
            db_user='postgres',
            db_password='1{Rithwik}',
            db_host='localhost',
            db_port='5432',
            db_name='postgres'
        )
        Xtr, Xv, Xt, ytr, yv, yt = split_data(df)

        # Keep raw for Evidently
        df_train = Xtr.copy()
        df_test  = Xt.copy()

        # Load or simulate new batch
        df_new = pd.read_csv("src\data\example_new_data.csv")

        df_train = df_train.dropna(axis=1, how='all')
        df_test = df_test.dropna(axis=1, how='all')
        df_new = df_new.dropna(axis=1, how='all')



        if "adjusted_total_usd" in df_new.columns:
            df_new = df_new.drop(columns=["adjusted_total_usd"])

        # Log Evidently reports
        log_evidently_report(df_train, df_test,      dataset_name="train_vs_test")
        log_evidently_report(df_train, df_new,        dataset_name="train_vs_new_batch")

        
# ---------- Main ----------
if __name__ == "__main__":
    
    # generate_all_drift_reports_with_mlflow(
    # train_path="data/train.csv",
    # test_path="data/test.csv",
    # new_path="data/new_data.csv",
    # target_column="adjusted_total_usd",
    # output_dir="drift_reports"
    # )

    # if drift_new > 0.5:
    #     print("Drift > 0.5 detected in train vs new. Triggering retraining pipeline.")
    #     trigger_airflow_dag()
    # else:
    #     print("No significant drift detected.")

    main()
