# PRACTICAL 3: Parsing and Processing of Semi-Structured JSON Data

# STEP 1: Open the terminal in the project folder
# STEP 2: Create a virtual environment
# python -m venv .venv
# STEP 3: Activate the environment
# CMD: .venv\Scripts\activate.bat
# PowerShell: .\.venv\Scripts\Activate.ps1
# STEP 4: Install required libraries
# python -m pip install requests pandas
# STEP 5: Run the program from the project folder
# python Practical3/practical3.py

import requests
import pandas as pd

url = "https://jsonplaceholder.typicode.com/users"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()

    records = []

    for user in data:
        address = user.get("address", {})
        geo = address.get("geo", {})
        company = user.get("company", {})

        record = {
            "User_ID": user.get("id"),
            "Name": user.get("name"),
            "Username": user.get("username"),
            "Email": user.get("email"),
            "Street": address.get("street"),
            "City": address.get("city"),
            "Zipcode": address.get("zipcode"),
            "Latitude": geo.get("lat"),
            "Longitude": geo.get("lng"),
            "Company": company.get("name")
        }

        records.append(record)

    df = pd.DataFrame(records)

    df["Latitude"] = pd.to_numeric(df["Latitude"])
    df["Longitude"] = pd.to_numeric(df["Longitude"])

    print("Transformed Dataset:")
    print(df)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns)

    print("\nData Types:")
    print(df.dtypes)

    print("\nUsers by City:")
    print(df["City"].value_counts())

    print("\nUser Names and Companies:")
    print(df[["Name", "Company"]])

    print("\nUsers from Gwenborough:")
    print(df[df["City"] == "Gwenborough"])

    print("\nSorted Users:")
    sorted_df = df.sort_values("Name")
    print(sorted_df[["User_ID", "Name", "City"]])

    print("\nSelected Columns:")
    selected_data = df[["User_ID", "Name", "Email", "City", "Company"]]
    print(selected_data)

    df.to_csv("Practical3/users_transformed.csv", index=False)

    print("\nFile saved successfully.")

    new_df = pd.read_csv("Practical3/users_transformed.csv")

    print("\nVerified CSV Data:")
    print(new_df.head())

    print("\nVerified Dataset Shape:")
    print(new_df.shape)

else:
    print("Error while retrieving data.")
    print("Status Code:", response.status_code)