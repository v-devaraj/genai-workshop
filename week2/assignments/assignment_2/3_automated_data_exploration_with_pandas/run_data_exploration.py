import pandas as pd
from automated_data_exploration import DataExplorer

# Load your dataset
df = pd.read_csv(r"C:\Users\devar\Documents\Dotkonnekt\genai-workshop\week2\data\titanic.csv")  # or pd.read_excel("file.xlsx")

# Create an instance of DataExplorer
explorer = DataExplorer(df)

# Generate the report
explorer.generate_report()