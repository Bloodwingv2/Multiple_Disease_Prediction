import pandas as pd  

# Load datasets
diabetes = pd.read_csv("data/diabetes.csv")
heart = pd.read_csv("data/heart.csv")
liver = pd.read_csv("data/liver.csv")

# Display first few rows
print("Diabetes Dataset:\n", diabetes.head(), "\n")
print("Heart Disease Dataset:\n", heart.head(), "\n")
print("Liver Disease Dataset:\n", liver.head(), "\n")

# Check for missing values
print("Missing Values:\n")
print("Diabetes:\n", diabetes.isnull().sum(), "\n")
print("Heart:\n", heart.isnull().sum(), "\n")
print("Liver:\n", liver.isnull().sum(), "\n")
