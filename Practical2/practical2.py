# PRACTICAL 2: Titanic Dataset Analysis

# STEP 1: Open the terminal in the project folder

# STEP 2: Create a virtual environment
# python -m venv .venv

# STEP 3: Activate the virtual environment
# CMD: .venv\Scripts\activate.bat
# PowerShell: .\.venv\Scripts\Activate.ps1

# STEP 4: Install required libraries
# python -m pip install pandas numpy matplotlib seaborn
# python -m pip install pandas numpy matplotlib seaborn scikit-learn requests
# STEP 5: Run the program
# python Practical2/practical2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Practical2/Titanic-Dataset.csv")

print("Titanic Dataset Loaded Successfully")

print("\nFirst 5 Rows:")
print(df.head())

print("\nFirst 3 Rows:")
print(df.head(3))

print("\nLast 5 Rows:")
print(df.tail())

print("\nLast 2 Rows:")
print(df.tail(2))

print("\nDataset Shape:")
print(df.shape)

print("\nNumber of Rows:")
print(df.shape[0])

print("\nNumber of Columns:")
print(df.shape[1])

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nDataset Index:")
print(df.index)

print("\nColumn Data Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentage:")
print(missing_percentage)

print("\nUnique Values in Each Column:")
print(df.nunique())

print("\nDataset Information:")
print("Total Entries:", len(df))
print("Total Columns:", len(df.columns))

print("\nAll Columns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nSurvival Counts:")
print(df["Survived"].value_counts())

print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)

print("\nPassenger Counts by Sex:")
print(df["Sex"].value_counts())

print("\nUnique Sex Values:")
print(df["Sex"].unique())

print("\nPassenger Class Counts:")
print(df["Pclass"].value_counts())

print("\nUnique Passenger Classes:")
print(df["Pclass"].unique())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass")
plt.title("Passenger Count by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex")
plt.title("Passenger Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Number of Passengers")
plt.show()
