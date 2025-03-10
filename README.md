# Multiple Disease Prediction Project ⚕️💉

This project leverages **Streamlit** to create a web-based application that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** based on user-provided health data.  

🟢 **Deployed at:** [https://hiremeplsthx.streamlit.app/]  

## Features
✅ Predicts multiple diseases using trained ML models  
✅ Displays model performance metrics (Accuracy, Precision, Recall, F1 Score)  
✅ Includes sample data for quick testing  
✅ Unified prediction interface with all three diseases accessible via the left sidebar  
✅ User-friendly interface powered by **Streamlit**  

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
### ⚙️ Create and Activate a Conda Environment
⚠️ Recommended: Creating a separate Conda environment helps isolate dependencies. However, you can skip this step if you're comfortable using global packages.
```bash
conda create -n disease_prediction_env python=3.10
conda activate disease_prediction_env
```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Select the desired disease prediction option from the sidebar.
2. Enter your health details in the provided input fields.
3. Click the **Predict** button to view the prediction result and model metrics.

## ✨ Key Enhancements
- ➕ Added **"Healthy"** and **"Non-Healthy"** buttons to simplify testing with pre-existing values.
- 🔽 Improved usability by integrating multiple diseases under a unified sidebar menu.
- 📊 Stored model performance metrics (accuracy, precision, etc.) in **JSON** format for easy data handling and streamlined updates.
- 📝 Utilized **Markdown** in Streamlit for a clearer and more informative presentation of results.
- 🧹 Enhanced code readability with well-structured comments and Markdown descriptions in Jupyter Notebook.

---

## 🔍 Future Enhancements
- 🔧 Perform a comprehensive code review to improve stability and performance.
- 📈 Experiment with models such as **Random Forest**, **SVM**, and **XGBoost** for enhanced prediction accuracy.
- 🎨 Improve the UI design to provide a better user experience.

---

## 🗂️ Project Structure (Refer Jupyter Notebook and run each cell)
The project follows a structured development pipeline:

1. **Environment Setup** ⚙️
2. **Dataset Acquisition** 📄
3. **Data Preprocessing** 🧪
4. **Model Training and Saving** 🧠
5. **Streamlit Deployment** 🌐
6. **Enhancements and Testing** ✅

---

## 📄 Dataset Acquisition

### 📚 Datasets Used
- **Pima Indians Diabetes Database** (Kaggle)  
- **Indian Parkinson's Patient Records** (Kaggle)  
- **Parkinson's Disease Dataset** (Kaggle)  

### 📥 Loading the Datasets
```python
# Load the datasets
diabetes = pd.read_csv("data/diabetes.csv")
heart = pd.read_csv("data/heart.csv")
parkinsons = pd.read_csv("data/parkinsons.csv")
```

### 💾 Saving Processed Data for Future Use
```python
# Save processed data
diabetes.to_csv("data/diabetes_cleaned.csv", index=False)
heart.to_csv("data/heart_cleaned.csv", index=False)
parkinsons.to_csv("data/parkinsons_cleaned.csv", index=False)
```

---

## 🧪 Data Preprocessing

- 🩺 Addressed **missing values**, **outliers**, and **scaling issues** for improved model performance.  
- 🔄 Utilized **`StandardScaler`** for consistent scaling to prevent skewed predictions.  

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

---

## 🧠 Train & Save the Models

### 🤖 Model Selection
- Implemented **Logistic Regression** for its simplicity and effectiveness.
- 🔍 Future plans include exploring **Random Forest**, **SVM**, and **XGBoost**.

### 💾 Training & Saving Models with Scalers
```python
from sklearn.linear_model import LogisticRegression
import joblib
import json

# Train and save the model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump({'model': model, 'scaler': scaler}, 'models/diabetes_model.pkl')

# Save performance metrics
metrics = {
    'accuracy': 0.89,
    'f1_score': 0.87,
    'recall': 0.85,
    'precision': 0.88
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
```

---

## 🌐 Streamlit Deployment

### 🧩 Key Features
- 🟢 **"Fill Healthy Values"** and 🔴 **"Fill Diabetic Values"** buttons simplify testing.
- 🔽 Implemented **dropdown menus** for binary options for improved user experience.
- 🔄 Utilized **session states** to manage dynamic inputs efficiently.
- ✅ Models are integrated with scalers to ensure accurate predictions in Streamlit.

### ▶️ Running the Application
```bash
streamlit run app.py
```

Visit the live app here: [https://hiremeplsthx.streamlit.app/]  

---

## ⭐ Contributing
If you find this project helpful, consider giving it a ⭐ and sharing your thoughts! Suggestions and improvements are welcome. 😊

---

## 📜 License
This project is licensed under the **MIT License**.

## 👨‍💻 Developed By
**Mirang Bhandari**

