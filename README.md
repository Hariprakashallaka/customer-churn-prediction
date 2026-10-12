# Customer Churn Prediction Using Machine Learning

## Project Overview

This project predicts whether a customer is likely to leave a service using a Machine Learning classification model. It uses the Random Forest Classifier and a Streamlit web application to provide predictions based on customer information.

The application allows users to enter customer details and receive a churn prediction through a simple, interactive interface.

## Objectives

* Build a customer churn prediction model using Random Forest.
* Perform data preprocessing using Scikit-learn pipelines.
* Evaluate model performance using classification metrics.
* Save the trained model as a `.pkl` file.
* Develop an interactive web application using Streamlit.
* Deploy the application online using Streamlit Community Cloud.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Streamlit
* Git and GitHub
* VS Code

## Project Structure

```text
customer-churn-streamlit/
│
├── app.py
├── train_model.py
├── churn_model.pkl
├── customer_churn_prediction.csv
├── requirements.txt
├── README.md
└── .gitignore
```

**File descriptions:**

* `app.py` — Creates the Streamlit interface and displays churn predictions.
* `train_model.py` — Loads the dataset, preprocesses the features, trains the model, evaluates performance, and saves the fitted pipeline.
* `churn_model.pkl` — Contains the saved preprocessing pipeline and trained Random Forest model.
* `customer_churn_prediction.csv` — Dataset used to train the model.
* `requirements.txt` — Lists the Python dependencies.
* `.gitignore` — Excludes unnecessary files from version control.

## Machine Learning Workflow

1. Load the customer churn dataset.
2. Separate the input features and target variable.
3. Remove the `CustomerID` identifier from the input features.
4. Identify numerical and categorical features.
5. Handle missing values using imputation.
6. Scale numerical features using StandardScaler.
7. Encode categorical features using OneHotEncoder.
8. Split the dataset into training and testing sets.
9. Train a Random Forest classification model.
10. Evaluate the model on unseen test data.
11. Save the complete fitted pipeline as `churn_model.pkl`.
12. Load the model in Streamlit and generate predictions.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-streamlit.git
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Navigate to the Project Folder

```bash
cd customer-churn-streamlit
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure that the Scikit-learn version is compatible with the version used to train the saved model.

## Train the Model

If you have the dataset and want to train the model again, run:

```bash
python train_model.py
```

This script trains the Random Forest pipeline, prints evaluation metrics, and creates the `churn_model.pkl` file.

The dataset must be available at the path configured in `train_model.py`.

## Run the Streamlit Application

Start the application locally using:

```bash
python -m streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Enter the customer details and click **Predict Churn** to see the prediction.

## Model Evaluation

The model is evaluated using the following metrics:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of predicted churn cases that are actual churn cases.
* **Recall:** Proportion of actual churn cases correctly identified.
* **F1-score:** Harmonic mean of precision and recall.
* **Confusion Matrix:** Shows correct and incorrect predictions for each class.
* **ROC-AUC:** Measures the model's ability to distinguish between the two classes when applicable.

Actual metric values should be taken from the output of `train_model.py` rather than assumed in advance.

## Deployment

The Streamlit application can be deployed using Streamlit Community Cloud.

1. Push the project files to GitHub.
2. Open https://share.streamlit.io/
3. Sign in using GitHub.
4. Create a new app from the existing repository.
5. Select the `main` branch.
6. Set the main file path to `app.py`.
7. Click **Deploy**.
8. Test the application using the generated public URL.

Ensure that `requirements.txt` contains the required packages and that the saved model is available in the repository.

## Important Notes

* The saved model and deployment environment must use compatible Scikit-learn versions.
* The input feature names and data types must match those used during training.
* `CustomerID` is excluded from model training because it is an identifier rather than a meaningful customer behavior feature.
* The dataset is required for retraining but not necessarily for running the deployed app.
* Avoid uploading confidential customer information, credentials, or API keys to GitHub.
* Predictions are estimates and should not be treated as guarantees of future customer behavior.

## Future Improvements

* Improve model performance through hyperparameter tuning.
* Add interactive charts and customer analytics.
* Compare Random Forest with Logistic Regression and Gradient Boosting.
* Add feature importance visualizations.
* Improve the user interface and prediction explanations.
* Monitor model performance as new customer data becomes available.

## Author

**Hari Prakash Allaka**

Machine Learning and Data Science Project

## License

This project is intended for educational and portfolio purposes. Add an appropriate open-source license if you intend to distribute or reuse the code publicly.
