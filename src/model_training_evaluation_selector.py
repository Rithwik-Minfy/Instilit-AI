# import os
# import joblib
# import numpy as np
# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# def train_evaluate_and_select_model(X_train, y_train, X_test, y_test, save_dir="pkl_joblib_files"):
#     os.makedirs(save_dir, exist_ok=True)

#     # Start MLflow experiment
#     mlflow.set_experiment("salary_prediction_experiment")

#     models = {
#         'ridge': {
#             'model': Ridge(),
#             'params': {'alpha': [0.1, 1.0, 10.0]}
#         },
#         'lasso': {
#             'model': Lasso(),
#             'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
#         },
#         'random_forest': {
#             'model': RandomForestRegressor(random_state=42),
#             'params': {
#                 'n_estimators': [100],
#                 'max_depth': [None, 10, 20],
#                 'min_samples_split': [2, 5]
#             }
#         },
#         'xgboost': {
#             'model': XGBRegressor(random_state=42),
#             'params': {
#                 'n_estimators': [100],
#                 'learning_rate': [0.05, 0.1],
#                 'max_depth': [3, 5]
#             }
#         },
#         'lightgbm': {
#             'model': LGBMRegressor(random_state=42),
#             'params': {
#                 'n_estimators': [100],
#                 'learning_rate': [0.05, 0.1],
#                 'max_depth': [-1, 5]
#             }
#         }
#     }

#     best_model = None
#     best_score = float('inf')
#     best_name = None

#     for name, config in models.items():
#         print(f"Training {name}...")

#         with mlflow.start_run(run_name=name):
#             grid = GridSearchCV(config['model'], config['params'],
#                                 cv=5, scoring='neg_root_mean_squared_error',
#                                 n_jobs=-1, verbose=0)
#             grid.fit(X_train, y_train)

#             rmse = -grid.best_score_
#             print(f"{name} best RMSE: {rmse:.4f} | Best Params: {grid.best_params_}")

#             # Log params & metrics
#             mlflow.log_params(grid.best_params_)
#             mlflow.log_metric("cv_rmse", rmse)

#             # Evaluate on test set
#             y_pred_test = grid.best_estimator_.predict(X_test)
#             test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
#             test_r2 = r2_score(y_test, y_pred_test)

#             mlflow.log_metric("test_rmse", test_rmse)
#             mlflow.log_metric("test_r2", test_r2)

#             # Log model
#             mlflow.sklearn.log_model(grid.best_estimator_, name)

#             if rmse < best_score:
#                 best_score = rmse
#                 best_model = grid.best_estimator_
#                 best_name = name
#                 best_y_pred = y_pred_test

#     print(f"\nBest Model: {best_name}")
#     print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, best_y_pred)):.4f}")
#     print(f"Test R² Score: {r2_score(y_test, best_y_pred):.4f}")

#     model_path = os.path.join(save_dir, "model.pkl")
#     joblib.dump(best_model, model_path)
#     print(f"Model saved to: {model_path}")

#     return best_model, best_y_pred


# if __name__ == "__main__":
#     best_model, y_pred_test = train_evaluate_and_select_model(X_train, y_train, X_test, y_test, save_dir="pkl_joblib_files")


import mlflow
import mlflow.sklearn
import os
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow


def train_evaluate_and_select_model(X_train, y_train, X_test, y_test, save_dir="transform_params", model_name="InstilitSalaryModel"):
    os.makedirs(save_dir, exist_ok=True)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("salary_prediction_experiment")


    models = {
        'ridge': {
            'model': Ridge(),
            'params': {'alpha': [0.1, 1.0, 10.0]}
        },
        'lasso': {
            'model': Lasso(),
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0]}
        },
        'random_forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'xgboost': {
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'lightgbm': {
            'model': LGBMRegressor(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [-1, 5]
            }
        }
    }

    best_model = None
    best_score = float('inf')
    best_name = None
    best_run_id = None

    for name, config in models.items():
        with mlflow.start_run(run_name=f"{name}_run") as run:
            print(f"Training {name}...")

            grid = GridSearchCV(config['model'], config['params'],
                                cv=5, scoring='neg_root_mean_squared_error',
                                n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)

            rmse_cv = -grid.best_score_
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("rmse_cv", rmse_cv)

            mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")

            if rmse_cv < best_score:
                best_score = rmse_cv
                best_model = grid.best_estimator_
                best_name = name
                best_run_id = run.info.run_id

    # Final evaluation on test data
    y_pred_test = best_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    print(f"Best Model: {best_name}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")

    # Save model locally for Flask
    model_path = os.path.join(save_dir, "model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Model saved to: {model_path}")

    # Register the best model from its run ID
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = mlflow.tracking.MlflowClient()

    # Move to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging",
        archive_existing_versions=True
    )
    print(f"Model version {result.version} moved to Staging")

    # Move to Production (optional/manual step; can skip if needed)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model version {result.version} moved to Production")

    return best_model, y_pred_test

if __name__=="__main__":
    best_model, y_pred_test = train_evaluate_and_select_model(X_train, y_train, X_test, y_test, save_dir="pkl_joblib_files")