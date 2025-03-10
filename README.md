# Multiple Disease Prediction Project

This project aims to develop a web application using Streamlit that predicts the likelihood of diabetes, heart disease, and liver disease based on user-provided health data.

## Project Structure

The project encompasses the following key steps:

1.  **Environment Setup**
2.  **Dataset Acquisition**
3.  **Data Preprocessing**
4.  **Model Training and Saving**
5.  **Streamlit Deployment**
6.  **Enhancements**

## Step 1: Environment Setup

### Create a Conda Environment

```bash
conda create -n disease_prediction_env python=3.10
conda activate disease_prediction_env
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

When using Jupyter Notebook in VS Code, ensure you select the newly created environment at the top. Restart VS Code if needed.

---

## Step 2: Dataset Acquisition

### Datasets Used

- **Pima Indians Diabetes Database** (from Kaggle)
- **Indian Liver Patient Records** (from Kaggle)
- **Parkinson's Disease** (from Kaggle)

### Load Datasets in Jupyter Notebook

```python
import pandas as pd

diabetes = pd.read_csv("path/to/diabetes.csv")
heart = pd.read_csv("path/to/heart.csv")
liver = pd.read_csv("path/to/liver.csv")
```

### Save Processed Data for Future Use

```python
diabetes.to_csv("diabetes_cleaned.csv", index=False)
heart.to_csv("heart_cleaned.csv", index=False)
liver.to_csv("liver_cleaned.csv", index=False)
```

---

## Step 3: Data Preprocessing

- Cleaned and processed data to address:
  - Missing values
  - Outliers
  - Scaling issues
    
- Applied `StandardScaler` to avoid bias from large values, as i was getting incorrect predictions.

---

## Step 4: Train & Save the Models

### Model Selection

- Started with **Logistic Regression** for simplicity and effectiveness, u can use anything, maybe another method.
- Future plans include exploring **Random Forest**, **SVM**, and **XGBoost**.

### Training & Saving Models

```python
from sklearn.linear_model import LogisticRegression
import joblib

model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'diabetes_model.pkl')
```

---

## Step 5: Streamlit Deployment

### Key Features

- **"Fill Healthy Values"** button for testing non-risk indicators.
- **"Fill Diabetic Values"** button for testing known diabetic indicators.
- Utilized dictionaries to manage reference values efficiently.
- Implemented session states for dynamic input management.

---

## Step 6: Enhancements

- Added "Healthy" and "Diabetic" buttons for the **Heart Disease** section as well.
- For fields with binary values (0/1), used index values for easier reference.
- Stored model metrics (accuracy, precision, etc.) in **JSON** format for efficient data handling.
- Displayed metrics in Streamlit using **Markdown** for improved clarity.
- Improved code readability with clear comments and Markdown descriptions in Jupyter Notebook.

---

## Next Steps

- Conduct a detailed code review to ensure stability and performance.
- Explore additional models like **Random Forest**, **SVM**, or **XGBoost** for performance comparison.
- Enhance the UI for improved user experience.

---

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/multiple-disease-prediction.git
   cd multiple-disease-prediction
   ```
2. **Install dependencies** using the instructions above.
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## Contact

For inquiries or contributions, feel free to reach out via **GitHub Issues**.

