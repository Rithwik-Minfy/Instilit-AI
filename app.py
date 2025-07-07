import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_file
from werkzeug.utils import secure_filename

# Load artifacts
MODEL_PATH = "src/pkl_joblib_files/model.pkl"
PREPROCESSOR_PATH = "src/pkl_joblib_files/preprocessor.pkl"
YEOJOHNSON_PATH = "src/pkl_joblib_files/yeojohnson_target_transformer.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Custom Yeo-Johnson wrapper
class YeoJohnsonTargetTransformer:
    def __init__(self):
        self.pt = None

    def load(self, path):
        self.pt = joblib.load(path)

    def inverse_transform(self, y_transformed):
        return self.pt.inverse_transform(np.array(y_transformed).reshape(-1, 1)).flatten()

yeojohnson = YeoJohnsonTargetTransformer()
yeojohnson.load(YEOJOHNSON_PATH)

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Expected input columns
input_columns = [
    'job_title', 'experience_level', 'employment_type', 'company_size', 'company_location',
    'remote_ratio', 'salary_currency', 'years_experience', 'base_salary', 'bonus',
    'stock_options', 'total_salary', 'salary_in_usd', 'currency', 'education',
    'skills', 'conversion_rate'
]

@app.route('/')
def index():
    return render_template('form.html', input_columns=input_columns)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            form_data = {col: request.form.get(col) for col in input_columns}

            # Convert numeric fields
            numeric_fields = ['remote_ratio', 'years_experience', 'base_salary', 'bonus',
                              'stock_options', 'total_salary', 'salary_in_usd', 'conversion_rate']
            for field in numeric_fields:
                form_data[field] = float(form_data[field]) if form_data[field] else 0.0

            df = pd.DataFrame([form_data])

            # Drop unneeded cols
            df.drop(columns=['education', 'skills'], errors='ignore', inplace=True)

            X_processed = preprocessor.transform(df)
            y_pred_trans = model.predict(X_processed)
            y_pred = yeojohnson.inverse_transform(y_pred_trans)

            return render_template('result.html', prediction=round(y_pred[0], 2))

        except Exception as e:
            return f"Error during prediction: {e}"

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)

            # Drop columns not required
            df.drop(columns=['education', 'skills'], errors='ignore', inplace=True)

            X_processed = preprocessor.transform(df)
            y_pred_trans = model.predict(X_processed)
            y_pred = yeojohnson.inverse_transform(y_pred_trans)

            df['Predicted_adjusted_total_usd'] = y_pred

            result_path = os.path.join(app.config['UPLOAD_FOLDER'], "prediction_results.csv")
            df.to_csv(result_path, index=False)

            return send_file(result_path, as_attachment=True)

        except Exception as e:
            return f"Error processing file: {e}"

@app.route('/download-template')
def download_template():
    sample_df = pd.DataFrame(columns=input_columns)
    template_path = os.path.join(app.config['UPLOAD_FOLDER'], "sample_template.csv")
    sample_df.to_csv(template_path, index=False)
    return send_file(template_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=8000)  
