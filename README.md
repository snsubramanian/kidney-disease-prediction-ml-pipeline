# Kidney Disease Prediction Using Machine Learning Pipeline

## Project Overview

This project focuses on predicting Chronic Kidney Disease (CKD) using Machine Learning techniques. It follows an end-to-end Machine Learning pipeline, starting from data collection and preprocessing to model training, evaluation, and deployment through a Streamlit web application.

The project uses a kidney disease dataset obtained from Kaggle. Different classification algorithms are trained and compared to identify a suitable model for the prediction task. Scikit-learn pipelines are used to combine data preprocessing and model training into a single workflow.

The Streamlit application allows users to enter patient-related information and view the model's prediction along with estimated class probabilities.

## Objectives

- Build an end-to-end Machine Learning pipeline for CKD classification.
- Clean and preprocess structured medical data.
- Handle missing numerical and categorical values.
- Apply feature scaling and categorical encoding.
- Train and compare multiple Machine Learning algorithms.
- Evaluate models using cross-validation metrics.
- Save the selected trained pipeline using Joblib.
- Develop an interactive prediction interface using Streamlit.

## Machine Learning Workflow

### 1. Data Collection
The kidney disease dataset is obtained from Kaggle and loaded into a Pandas DataFrame.

### 2. Data Cleaning
- Standardize column names and text values.
- Replace invalid or missing-value representations with `NaN`.
- Convert numerical columns to appropriate data types.
- Remove the unnecessary ID column.
- Remove records with missing target labels.

### 3. Data Preprocessing
- Use median imputation for missing numerical values.
- Use most-frequent-value imputation for categorical features.
- Apply `StandardScaler` to numerical features.
- Apply `OneHotEncoder` to categorical features.
- Combine preprocessing steps using `ColumnTransformer`.

### 4. Train-Test Split
Split the dataset into training and testing sets using an 80:20 ratio, with stratification to preserve the target-class distribution.

### 5. Model Training and Comparison
The following classification algorithms are compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Gradient Boosting Classifier
- AdaBoost Classifier

### 6. Model Evaluation
Use five-fold Stratified Cross-Validation to compare models using:

- Accuracy
- Precision
- Recall
- F1-score

The models are ranked by their mean cross-validation F1-score, and the top-ranked model is selected for further use. The held-out test set can be used for final evaluation.

### 7. Model Saving
Save the fitted preprocessing and classification pipeline using Joblib so that the same transformations are applied when making predictions in the Streamlit application.

### 8. Web Application
Build an interactive web interface using Streamlit. Users can enter input values and view the predicted class and estimated probabilities.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Joblib
- **Web Application:** Streamlit
- **Development Environment:** Jupyter Notebook, Visual Studio Code

## Project Structure

```text
kidney-disease-prediction-ml-pipeline/
│
├── app.py
├── mlpipe.ipynb
├── kidney_disease.csv
├── kidney_model_comparison.csv
├── kidney_disease_best_pipeline.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The dataset and trained model can be excluded from GitHub depending on the dataset license, file size, and deployment requirements.*

## Installation and Setup

### Prerequisites
- Python installed on your system
- Git
- A compatible Python environment for the installed Scikit-learn version

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/kidney-disease-prediction-ml-pipeline.git
```

### 2. Navigate to the Project Folder

```bash
cd kidney-disease-prediction-ml-pipeline
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Required Files

Ensure the following files are available in the project directory:

- `kidney_disease_best_pipeline.pkl`
- Any dataset file required to rerun the notebook

The Streamlit application needs the saved model file to make predictions.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser. Follow the instructions on the page to enter input values and view the prediction.

## Results

The notebook compares nine classification algorithms using five-fold Stratified Cross-Validation. The comparison results are saved in `kidney_model_comparison.csv`.

The selected model is determined by the mean cross-validation F1-score. Actual performance values should be taken from the notebook's evaluation results rather than assumed in advance.

## Future Improvements

- Add more detailed model evaluation visualizations.
- Improve the Streamlit interface and input validation.
- Evaluate the selected model on an independent test set.
- Add reproducible training and model-versioning steps.
- Explore hyperparameter tuning to improve model performance.

## Disclaimer

This project is intended for educational and machine learning demonstration purposes only. The predictions are not medical diagnoses and should not be used to make clinical decisions. Always consult a qualified healthcare professional for medical advice.

## Author

**Subramanian S N**

GitHub: https://github.com/snsubramanian
