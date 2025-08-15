import pandas as pd
import numpy as np

class DataExplorer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataExplorer with a pandas DataFrame.
        """
        self.df = df

    def statistical_summary(self):
        """
        Generates statistical summaries for each column.
        """
        return self.df.describe(include='all').transpose()

    def detect_missing_values(self):
        """
        Detects missing values in the DataFrame.
        """
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        return pd.DataFrame({
            'Missing Values': missing_count,
            'Missing %': missing_percent
        })

    def detect_duplicates(self):
        """
        Detects duplicate rows in the DataFrame.
        """
        duplicate_count = self.df.duplicated().sum()
        return duplicate_count

    def detect_outliers(self):
        """
        Detects outliers using the IQR method for numeric columns.
        """
        outlier_summary = {}
        numeric_cols = self.df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_summary[col] = outliers

        return pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier Count'])

    def recommend_preprocessing(self):
        """
        Suggests preprocessing steps based on the dataset issues.
        """
        recommendations = []

        # Missing value handling
        missing = self.detect_missing_values()
        for col, row in missing.iterrows():
            if row['Missing Values'] > 0:
                recommendations.append(f"Column '{col}': Handle {row['Missing Values']} missing values (e.g., imputation or removal).")

        # Duplicate handling
        duplicates = self.detect_duplicates()
        if duplicates > 0:
            recommendations.append(f"Dataset has {duplicates} duplicate rows — consider removing them.")

        # Outlier handling
        outliers = self.detect_outliers()
        for col, row in outliers.iterrows():
            if row['Outlier Count'] > 0:
                recommendations.append(f"Column '{col}': {row['Outlier Count']} potential outliers — consider capping or transformation.")

        if not recommendations:
            recommendations.append("No major preprocessing steps needed.")

        return recommendations

    def generate_report(self):
        """
        Generates and prints a structured data exploration report.
        """
        print("\n===== DATA EXPLORATION REPORT =====\n")

        print("1. Statistical Summary:\n")
        print(self.statistical_summary(), "\n")

        print("2. Missing Values:\n")
        print(self.detect_missing_values(), "\n")

        print("3. Duplicate Rows:\n")
        print(f"Total Duplicates: {self.detect_duplicates()}\n")

        print("4. Outliers:\n")
        print(self.detect_outliers(), "\n")

        print("5. Recommended Preprocessing Steps:\n")
        for step in self.recommend_preprocessing():
            print(f"- {step}")

        print("\n===== END OF REPORT =====\n")


# =============================
# Example Usage
# =============================
if __name__ == "__main__":
    # Example DataFrame
    data = {
        'A': [1, 2, 2, 4, 100, np.nan],
        'B': [5, 5, 5, 5, 5, 5],
        'C': ['x', 'y', 'y', np.nan, 'z', 'z']
    }
    df = pd.DataFrame(data)

    explorer = DataExplorer(df)
    explorer.generate_report()