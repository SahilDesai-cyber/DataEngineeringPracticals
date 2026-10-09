# PRACTICAL 4: Data Acquisition from REST APIs Using Python

# STEP 1: Open the terminal in the project folder
# STEP 2: Create a virtual environment
# python -m venv .venv
# STEP 3: Activate the environment
# CMD: .venv\Scripts\activate.bat
# PowerShell: .\.venv\Scripts\Activate.ps1
# STEP 4: Install required libraries
# python -m pip install requests pandas
# STEP 5: Run the program from the project folder
# python Practical4/practical4.py

import requests
import pandas as pd
import time

print("=" * 65)
print("DATA ACQUISITION FROM REST APIs USING PYTHON")
print("=" * 65)

url = "https://jsonplaceholder.typicode.com/users"

print("\nAPI Endpoint:", url)

start = time.time()
response = requests.get(url)
end = time.time()

print("\nStatus Code:", response.status_code)
print("Response Time:", end - start, "seconds")

if response.status_code == 200:
    data = response.json()

    print("\nJSON data received successfully.")
    print("Total records:", len(data))

    print("\nFirst record from API:")
    print(data[0])

    df = pd.json_normalize(data)

    print("\nFirst 5 rows of the DataFrame:")
    print(df[["id", "name", "username", "email", "phone"]].head().to_string(index=False))

    print("\nTotal rows:", df.shape[0])
    print("Total columns:", df.shape[1])

    output_file = "Practical4/users_data.csv"
    df.to_csv(output_file, index=False)

    print("\nUsers CSV saved successfully.")
    print("Filename:", output_file)

    verify = pd.read_csv(output_file)

    print("\nVerifying saved CSV:")
    print("Rows:", verify.shape[0])
    print("Columns:", verify.shape[1])

    posts_url = "https://jsonplaceholder.typicode.com/posts?userId=1"

    posts_response = requests.get(posts_url)

    print("\nSecond API Status Code:", posts_response.status_code)

    if posts_response.status_code == 200:
        posts_data = posts_response.json()
        posts_df = pd.json_normalize(posts_data)

        print("Total posts by User 1:", len(posts_df))

        print("\nFirst 3 posts by User 1:")
        print(posts_df[["userId", "id", "title"]].head(3).to_string(index=False))

        posts_df.to_csv("Practical4/posts_data.csv", index=False)

        print("\nPosts CSV saved successfully.")
        print("Filename: Practical4/posts_data.csv")

    else:
        print("Unable to retrieve posts.")

else:
    print("Error while retrieving user data.")
    print("Status Code:", response.status_code)

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED")
print("=" * 65)