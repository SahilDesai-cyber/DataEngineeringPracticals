# PRACTICAL 5: DATA QUALITY ASSESSMENT, CLEANING, AND TRANSFORMATION
#
# WINDOWS SETUP INSTRUCTIONS
#
# STEP 1: Open the terminal in the project folder
#
# STEP 2: Create a virtual environment (only on Windows)
# python -m venv .venv
#
# STEP 3: Activate the environment
# CMD: .venv\Scripts\activate.bat
# PowerShell: .\.venv\Scripts\Activate.ps1
#
# STEP 4: Install required libraries
# python -m pip install pandas numpy matplotlib seaborn scikit-learn
#
# STEP 5: Run the program from the project folder
# python Practical5/practical5.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("Libraries imported successfully.")
print("Pandas version:", pd.__version__)

url = "https://howtocsv.com/samples/sampledirtydata.csv"

raw = pd.read_csv(url, keep_default_na=False)

print("\nRaw dataset loaded.")
print("Shape:", raw.shape)
print("\nColumn names:", list(raw.columns))
print("\nData types:\n", raw.dtypes)
print("\nFirst 5 rows:\n", raw.head())
print("\nLast 5 rows:\n", raw.tail())
print("\nStatistical summary:\n", raw.describe(include="all").T)
raw.info()

missing_report = pd.DataFrame({
    "Empty_String": (raw == "").sum(),
    "Literal_NA": (raw == "N/A").sum(),
    "Total_Percent": (
        ((raw == "") | (raw == "N/A")).sum() / len(raw) * 100
    ).round(2)
}).sort_values("Total_Percent", ascending=False)

print("\nMissing value report:")
print(missing_report[missing_report["Total_Percent"] > 0])

print("\nFully duplicated rows:", raw.duplicated().sum())
print(
    "Duplicated Full Name values:",
    raw["Full Name"].duplicated().sum()
)

print("\nDuplicate records:")
print(
    raw[raw.duplicated(keep=False)]
    .sort_values("Full Name")
)

for col in ["country", "is_active"]:
    print(f"\nUnique values in {col}:")
    print(sorted(raw[col].unique()))

print("\nSignup date values:", raw["signup date"].unique())
print("\nAmount values:", raw["amount"].unique())
print("\nSample email values:", raw["email"].unique()[:6])

email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

invalid_emails = raw[
    ~raw["email"].astype(str).str.strip().str.match(
        email_pattern, na=False
    )
]

print("\nInvalid email records:")
print(invalid_emails[["id", "Full Name", "email"]])

temp_amount = (
    raw["amount"]
    .astype(str)
    .str.replace(r"[\$,€]", "", regex=True)
    .str.replace(",", "", regex=False)
    .str.replace("N/A", "NaN", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

def find_outliers_iqr(series):
    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]

    return lower, upper, outliers


low, up, out = find_outliers_iqr(temp_amount)

print(f"\nLower bound = {low:.2f} | Upper bound = {up:.2f}")
print("Number of outliers detected:", len(out))
print(out)

plt.figure(figsize=(7, 4))
sns.boxplot(x=temp_amount.dropna())
plt.title("Boxplot of Amount Before Cleaning")
plt.tight_layout()
plt.show()


def quality_report(data, label):
    name_col = "full_name" if "full_name" in data.columns else "Full Name"

    missing_count = (
        (data == "").sum().sum()
        + (data == "N/A").sum().sum()
    )

    return pd.Series({
        "Dataset": label,
        "Rows": data.shape[0],
        "Columns": data.shape[1],
        "Missing Cells (blank/N/A)": int(missing_count),
        "Duplicate Rows": int(data.duplicated().sum()),
        "Duplicate Names": int(data[name_col].duplicated().sum()),
        "Object-type Columns": int(
            (data.dtypes == "object").sum()
        )
    })


before_report = quality_report(raw, "BEFORE Cleaning")

print("\nQuality report before cleaning:")
print(before_report.to_frame().T)

df = raw.copy()

rows_before = df.shape[0]

df = df.drop_duplicates().reset_index(drop=True)

print(
    f"\nRemoved {rows_before - df.shape[0]} exact duplicate row(s)."
)
print("New shape:", df.shape)

df["Full Name"] = (
    df["Full Name"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
    .str.title()
)

df["email"] = (
    df["email"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("\nSample cleaned names:", df["Full Name"].head().tolist())
print("Sample cleaned emails:", df["email"].head().tolist())

dupes = df[df.duplicated(subset=["Full Name"], keep=False)]

print("\nRows with duplicate names after normalisation:")
print(dupes[["id", "Full Name", "email"]])

df = (
    df.drop_duplicates(subset=["Full Name"], keep="first")
    .reset_index(drop=True)
)

print("Shape after removing duplicate names:", df.shape)

country_map = {
    "usa": "US",
    "u.s.a.": "US",
    "united states": "US",
    "china": "CN",
    "spain": "ES",
    "de": "DE",
    "germany": "DE"
}

df["country"] = df["country"].astype(str).str.strip().str.lower()

df["country"] = df["country"].replace({
    "": np.nan,
    "n/a": np.nan,
    "nan": np.nan
})

df["country"] = (
    df["country"]
    .map(country_map)
    .fillna(df["country"].str.upper())
)

print("\nCleaned country values:", sorted(df["country"].dropna().unique()))

truthy = {"yes", "y", "true", "1", "active"}
falsy = {"no", "n", "false", "0"}

def parse_bool(value):
    value = str(value).strip().lower()

    if value in truthy:
        return True

    if value in falsy:
        return False

    return np.nan


df["is_active"] = df["is_active"].apply(parse_bool)

print("\nCleaned is_active values:")
print(df["is_active"].value_counts(dropna=False))


def parse_mixed_date(value):
    if pd.isna(value) or str(value).strip() in ("", "N/A"):
        return pd.NaT

    value = str(value).strip()

    formats = [
        "%Y%m%d",
        "%d/%m/%Y",
        "%b %d, %Y",
        "%Y/%m/%d",
        "%Y.%m.%d"
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue

    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        return pd.NaT


df["signup_date"] = df["signup date"].apply(parse_mixed_date)

print("\nSignup date data type:", df["signup_date"].dtype)
print("Unparsed dates:", df["signup_date"].isnull().sum())
print(df[["signup date", "signup_date"]].head(10))


def parse_amount(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.upper() in ("N/A", "", "FREE"):
        return np.nan

    value = value.replace("$", "").replace("€", "").replace(" ", "")

    if "." in value and "," in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")

    elif "," in value:
        if len(value.split(",")[-1]) == 2:
            value = value.replace(",", ".")
        else:
            value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return np.nan


df["amount"] = df["amount"].apply(parse_amount)

print("\nAmount data type:", df["amount"].dtype)
print("Sample cleaned amounts:", df["amount"].head(10).tolist())
print("Missing amounts:", df["amount"].isnull().sum())

df["email_valid"] = df["email"].str.match(email_pattern, na=False)

print("\nInvalid emails found:", (~df["email_valid"]).sum())

df.loc[
    ~df["email_valid"], "email"
] = "not.provided@unknown.com"

df["signup_date"] = df["signup_date"].fillna(
    df["signup_date"].max()
)

df["amount"] = df["amount"].fillna(
    df["amount"].median()
)

country_mode = df["country"].mode()

if not country_mode.empty:
    df["country"] = df["country"].fillna(country_mode.iloc[0])
else:
    df["country"] = df["country"].fillna("UNKNOWN")

active_mode = df["is_active"].mode()

if not active_mode.empty:
    df["is_active"] = df["is_active"].fillna(active_mode.iloc[0])
else:
    df["is_active"] = df["is_active"].fillna(False)

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("Total missing cells:", df.isnull().sum().sum())

low, up, out = find_outliers_iqr(df["amount"])

print(f"\nCapping amount to [{low:.2f}, {up:.2f}]")

before_max = df["amount"].max()

df["amount"] = df["amount"].clip(lower=low, upper=up)

print("Maximum before capping:", before_max)
print("Maximum after capping:", df["amount"].max())

plt.figure(figsize=(7, 4))
sns.boxplot(x=df["amount"])
plt.title("Boxplot of Amount After Outlier Treatment")
plt.tight_layout()
plt.show()

df.drop(
    columns=["signup date", "email_valid"],
    inplace=True,
    errors="ignore"
)

df.columns = [
    col.strip().lower().replace(" ", "_")
    for col in df.columns
]

today = pd.Timestamp.today().normalize()

df["signup_year"] = df["signup_date"].dt.year
df["signup_month"] = df["signup_date"].dt.month_name()

df["tenure_days"] = (
    today - df["signup_date"]
).dt.days

region_map = {
    "US": "North America",
    "DE": "Europe",
    "ES": "Europe",
    "CN": "Asia",
    "FR": "Europe",
    "GB": "Europe"
}

df["region"] = df["country"].map(region_map).fillna("Other")

print("\nEngineered features:")
print(
    df[
        [
            "id", "full_name", "signup_date",
            "signup_year", "tenure_days",
            "country", "region", "amount"
        ]
    ].head()
)

df["is_active_encoded"] = df["is_active"].astype(int)

df = pd.get_dummies(
    df,
    columns=["country"],
    prefix="country",
    dtype=int
)

df = pd.get_dummies(
    df,
    columns=["region"],
    prefix="region",
    dtype=int
)

print("\nColumns after categorical encoding:")
print(list(df.columns))

minmax = MinMaxScaler()
standard = StandardScaler()

df["amount_minmax"] = minmax.fit_transform(
    df[["amount"]]
).ravel()

df["tenure_days_zscore"] = standard.fit_transform(
    df[["tenure_days"]]
).ravel()

print("\nScaled numerical values:")
print(
    df[
        [
            "amount", "amount_minmax",
            "tenure_days", "tenure_days_zscore"
        ]
    ].head()
)

key_cols = [
    "id", "full_name", "email",
    "signup_date", "amount", "is_active"
]

other_cols = [col for col in df.columns if col not in key_cols]

df = df[key_cols + other_cols]

print("\nFinal dataset shape:", df.shape)
print("\nFinal dataset preview:")
print(df.head())

print("\n" + "=" * 55)
print("FINAL DATA QUALITY VALIDATION")
print("=" * 55)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Total missing cells:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate full names:", df["full_name"].duplicated().sum())
print("Data types:\n", df.dtypes.value_counts())
print("=" * 55)

assert df.isnull().sum().sum() == 0, "Missing values still present!"
assert df.duplicated().sum() == 0, "Duplicate rows still present!"
assert df["full_name"].is_unique, "Full names are not unique!"
assert df["amount"].between(0, 1e6).all(), "Invalid amount values!"
assert df["is_active"].isin([True, False]).all(), "Invalid active values!"

print("\nAll validation assertions PASSED. Dataset is reliable.")

after_report = quality_report(df, "AFTER Cleaning")

comparison = pd.concat(
    [before_report, after_report],
    axis=1
).T

print("\nBefore and after quality comparison:")
print(comparison)

raw_missing = (
    raw.eq("") | raw.eq("N/A")
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.heatmap(
    raw_missing,
    cbar=False,
    cmap="viridis",
    yticklabels=False,
    ax=axes[0]
)

axes[0].set_title("BEFORE Cleaning - Missing Values")

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis",
    yticklabels=False,
    ax=axes[1]
)

axes[1].set_title("AFTER Cleaning - Missing Values")

plt.tight_layout()
plt.show()

output_file = "Practical5/cleaned_data.csv"

df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved as '{output_file}'")
print(
    "File size:",
    round(len(df.to_csv(index=False).encode("utf-8")) / 1024, 2),
    "KB"
)