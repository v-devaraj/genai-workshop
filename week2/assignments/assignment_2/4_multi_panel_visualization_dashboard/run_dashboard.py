import pandas as pd
from multi_panel_dashboard import MultiPanelDashboard

# Step 1: Load your dataset
df = pd.read_csv(r"C:\Users\devar\Documents\Dotkonnekt\genai-workshop\week2\data\iris.csv")  # or pd.read_excel("file.xlsx")  # or pd.read_excel("your_file.xlsx")

# Step 2: Create dashboard object
dashboard = MultiPanelDashboard(df)

# Step 3: Generate the dashboard
dashboard.plot_dashboard()