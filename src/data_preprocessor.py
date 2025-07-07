import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PowerTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize

# Winsorization utility
def winsorize_columns(df, columns, limits=(0.01, 0.01)):
    for col in columns:
        try:
            df[col] = winsorize(df[col], limits=limits)
        except Exception as e:
            print(f"Could not winsorize column '{col}': {e}")
    return df

# Custom Yeo-Johnson transformer for target
class YeoJohnsonTargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, y):
        y = np.array(y).reshape(-1, 1)
        self.pt.fit(y)
        return self

    def transform(self, y):
        y = np.array(y).reshape(-1, 1)
        return self.pt.transform(y).flatten()

    def inverse_transform(self, y_transformed):
        y_transformed = np.array(y_transformed).reshape(-1, 1)
        return self.pt.inverse_transform(y_transformed).flatten()

    def save(self, path):
        joblib.dump(self.pt, path)

    def load(self, path):
        self.pt = joblib.load(path)

# Main preprocessing function
def preprocess_data(df, save_dir="pkl_joblib_files"):
    os.makedirs(save_dir, exist_ok=True)

    target_col = 'adjusted_total_usd'
    numeric_cols = ['years_experience', 'base_salary', 'bonus', 'stock_options', 'total_salary', 'salary_in_usd']
    categorical_cols = ['salary_currency', 'currency']
    ordinal_cols = ['experience_level', 'company_size']
    ordinal_map = [
        ['Junior', 'Mid', 'Senior', 'Lead'],
        ['Small', 'Medium', 'Large']
    ]

    # Step 1: Split data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 2: Winsorize
    X_train = winsorize_columns(X_train.copy(), numeric_cols)
    X_test = winsorize_columns(X_test.copy(), numeric_cols)

    # Step 3: Target transformation (Yeo-Johnson)
    y_transformer = YeoJohnsonTargetTransformer()
    y_transformer.fit(y_train)
    y_train_trans = y_transformer.transform(y_train)
    y_test_trans = y_transformer.transform(y_test)

    # Save the target transformer
    y_transformer.save(os.path.join(save_dir, "yeojohnson_target_transformer.pkl"))

    # Step 4: Column setup
    ordinal_features = [col for col in ordinal_cols if col in X.columns]
    ordinal_ordering = [ordering for col, ordering in zip(ordinal_cols, ordinal_map) if col in X.columns]
    nominal_features = [col for col in categorical_cols if col not in ordinal_features]

    # Step 5: Build transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('yeojohnson', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_ordering))
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Step 6: Combine all
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('ord', ordinal_transformer, ordinal_features),
        ('nom', nominal_transformer, nominal_features)
    ])

    # Step 7: Transform X
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    #  Save the full preprocessor
    joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.pkl"))

    # Step 8: Rebuild DataFrames
    encoded_nominal_cols = preprocessor.named_transformers_['nom']['onehot'].get_feature_names_out(nominal_features)
    feature_names = numeric_cols + ordinal_features + list(encoded_nominal_cols)

    X_train_df = pd.DataFrame(X_train_trans, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_trans, columns=feature_names, index=X_test.index)

    print(" Preprocessing completed and saved:")
    print(f" Preprocessor: {save_dir}/preprocessor.pkl")
    print(f" Yeo-Johnson for y: {save_dir}/yeojohnson_target_transformer.pkl")
    print(" X_train shape:", X_train_df.shape)
    print(" y_train (transformed) shape:", y_train_trans.shape)

    return X_train_df, X_test_df, y_train_trans, y_test_trans, y_transformer


if __name__=="__main__":
    X_train, X_test, y_train, y_test, y_transformer = preprocess_data(df_cleaned)