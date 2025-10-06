This code builds a machine learning pipeline to predict whether a patient will be readmitted to the hospital within 30 days using the UCI Diabetes dataset. It includes preprocessing, model training, calibration evaluation, feature importance, and clinical utility analysis.

 1. Data Loading and Preprocessing

df = pd.read_csv("diabetic_data.csv")
df.drop([...], axis=1, inplace=True)
df.replace('?', pd.NA, inplace=True)


- Loads the dataset and removes irrelevant or high-missing columns.
- Replaces placeholder '?' with proper missing value markers (NaN).

 2. Categorical Encoding

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


- Converts all categorical columns into numeric format using label encoding.
- This is necessary for ML models like Random Forest and XGBoost.

 3. Missing Value Imputation

imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


- Fills missing values using the most frequent value in each column.
- Fast and simple, suitable for low-end systems.

 4. Target Variable Transformation

df_imputed['readmitted'] = df_imputed['readmitted'].apply(lambda x: 1 if x == le.transform(['<30'])[0] else 0)


- Converts the target column readmitted into binary:
- 1 if readmitted within 30 days (<30)
- 0 otherwise

 5. Feature Selection

X = df_imputed.drop('readmitted', axis=1)
y = df_imputed['readmitted']
X_selected = VarianceThreshold(threshold=0.01).fit_transform(X)


- Removes features with very low variance (i.e., not informative).
- Helps reduce dimensionality and overfitting.

 6. Feature Scaling and Train/Test Split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
X_train, X_test, y_train, y_test = train_test_split(...)


- Standardizes features to have mean 0 and variance 1.
- Splits data into training and testing sets (80/20), stratified by target.

 7. Model Training

rf = RandomForestClassifier(...)
xgb = XGBClassifier(...)
lr = LogisticRegression(...)


- Trains three models:
- Random Forest (ensemble of decision trees)
- XGBoost (gradient boosting)
- Logistic Regression (baseline linear model)

 8. Calibration and Brier Score Evaluation

calibration_curve(...)
brier_score_loss(...)


- Plots calibration curves to show how well predicted probabilities match actual outcomes.
- Brier score quantifies probability accuracy (lower is better).

 9. Feature Importance (Random Forest)

importances = rf.feature_importances_


- Displays top 10 most important features used by Random Forest.
- Helps clinicians understand which variables drive predictions.

 10. Clinical Utility Analysis

confusion_matrix(...)
classification_report(...)


- Evaluates model predictions using:
- Confusion matrix (TP, FP, FN, TN)
- Precision, recall, F1-score
- Helps assess how well the model identifies high-risk patients.

 Summary
This pipeline:
- Handles mixed data types and missing values
- Trains and compares multiple models
- Evaluates calibration and clinical impact
- Provides interpretable insights for healthcare professionals
