# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Loading CSV file into a dataframe and showing the first 10 rows
# Note: Titanic dataset used here is downloaded from kaggle dataset repository
df = pd.read_csv(r"C:\Users\devar\Documents\Dotkonnekt\genai-workshop\week2\data\titanic.csv")
df.head()

# Displaying shape, column names, and data types.
print("Shape of the DataFrame:", df.shape)
print("Column Names:", df.columns.tolist())
print("Data Types:\n", df.dtypes)

# Description of the dataset
print("Summary Statistics:\n", df.describe())

# Count missing values per column.
print("Missing Values:\n", df.isna().sum())


# Fill missing numeric values with the mean.
for column in df.columns.tolist():
    if df[column].isna().sum() > 0 and df[column].dtype in ['float64', 'int64']:
        df[column].fillna(df[column].mean(), inplace=True)


print("Missing Values:\n", df.isna().sum())

# Filter rows by a numeric condition (e.g., Age > 30)
filtered_df = df[df['Age'] > 30]
print("Filtered DataFrame (Age > 30):\n", filtered_df.head())

# Sort dataset by a column in descending order (eg. PassengerId).
sorted_df = df.sort_values(by = 'PassengerId', ascending = False)
print("Sorted DataFrame by PassengerId (Descending):\n", sorted_df.head())


# Group by a categorical column, calculate mean of a numeric column ()
grouped_df = df.groupby("Sex")["Age"].mean().reset_index()
print("Grouped DataFrame (Mean Age):\n", grouped_df)


# Create a histogram for a numeric column
plt.hist(x=df['Age'].dropna(), bins=20, color='green', edgecolor='black')  # dropna to avoid NaN issues

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Create a bar chart of group averages
plt.bar(grouped_df["Sex"], grouped_df["Age"], color=["green", "lightblue"])
plt.title("Mean Age by Sex")
plt.xlabel("Sex")
plt.ylabel("Mean Age")
plt.show()


# Save the cleaned dataset as processed_data.csv
df.to_csv(r"C:\Users\devar\Documents\Dotkonnekt\genai-workshop\week2\data\processed_titanic.csv", index=False)

