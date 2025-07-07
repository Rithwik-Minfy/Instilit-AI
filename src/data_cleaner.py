def clean_data(df):
    # Clean salary columns by removing leading single quotes and converting to numeric
    salary_columns = ['base_salary', 'total_salary', 'salary_in_usd', 'adjusted_total_usd']
    for col in salary_columns:
        try:
            # Remove leading single quote and convert to float
            df[col] = df[col].astype(str).str.lstrip("'").astype(float)
        except ValueError as e:
            raise ValueError(f"Invalid values in {col}: unable to convert to numeric after removing single quotes. Error: {e}")
        
    # Remove education and skills columns if they exist
    df = df.drop(columns=['education', 'skills'], errors='ignore')

    # many duplicate rows found so dropping
    df.drop_duplicates()

    # Compute modes
    exp_mode = df['experience_level'].mode().iloc[0]
    emp_mode = df['employment_type'].mode().iloc[0]

    # Fill missing/unknown values with mode
    df['experience_level'].fillna(exp_mode, inplace=True)
    df['employment_type'].fillna(emp_mode, inplace=True)

    print(f" experience_level mode used: {exp_mode}")
    print(f" employment_type mode used: {emp_mode}")


    # Mapping inconsistent job titles to standard ones
    job_title_mapping = {
        'Software Engr': 'Software Engineer',
        'Sofware Engneer': 'Software Engineer',
        'Softwre Engineer': 'Software Engineer',
        
        'Data Scienist': 'Data Scientist',
        'Data Scntist': 'Data Scientist',
        'Dt Scientist': 'Data Scientist',
        
        'ML Engr': 'Machine Learning Engineer',
        'Machine Learning Engr': 'Machine Learning Engineer',
        'ML Enginer': 'Machine Learning Engineer',
        'ML Engineer': 'Machine Learning Engineer'
    }

    # Apply the mapping
    df['job_title'] = df['job_title'].replace(job_title_mapping)

    return df


if __name__ == "__main__":
    df_cleaned = clean_data(df)