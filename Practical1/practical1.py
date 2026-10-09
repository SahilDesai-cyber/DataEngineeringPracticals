# STEP 1: Open the terminal in the project folder

# STEP 2: Create a virtual environment
# python -m venv .venv

# STEP 3: Activate the environment
# CMD: .venv\Scripts\activate.bat
# PowerShell: .\.venv\Scripts\Activate.ps1

# STEP 4: Install the required library
# python -m pip install pandas

# STEP 5: Run this program from the project folder
# python Practical1/practical1.py

# STEP 6: Import pandas

import pandas as pd

df = pd.read_csv("Practical1/studentdata.csv")

print("Student Data:")
print(df)

print("\nFirst 3 Students:")
print(df.head(3))

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Information:")
print(df.describe())

print("\nStudents by Gender:")
print(df["Gender"].value_counts())

print("\nAverage Marks:", df["Marks"].mean())
print("Highest Marks:", df["Marks"].max())
print("Lowest Marks:", df["Marks"].min())