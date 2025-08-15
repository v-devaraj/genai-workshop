import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class MultiPanelDashboard:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_style("whitegrid")  # Clean look

    def plot_dashboard(self):
        """
        Automatically generates a multi-panel visualization dashboard.
        """
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        plot_idx = 0

        # Numeric: Histogram + KDE
        if num_cols:
            col = num_cols[0]
            sns.histplot(self.df[col], kde=True, ax=axes[plot_idx], color='skyblue')
            axes[plot_idx].set_title(f"Distribution of {col}")
            axes[plot_idx].set_xlabel(col)
            plot_idx += 1

        # Numeric: Boxplot for Outliers
        if len(num_cols) > 1:
            col = num_cols[1]
            sns.boxplot(x=self.df[col], ax=axes[plot_idx], color='lightgreen')
            axes[plot_idx].set_title(f"Boxplot of {col}")
            plot_idx += 1

        # Categorical: Count Plot
        if cat_cols:
            col = cat_cols[0]
            sns.countplot(x=self.df[col], ax=axes[plot_idx], palette="Set2")
            axes[plot_idx].set_title(f"Category Counts: {col}")
            axes[plot_idx].tick_params(axis='x', rotation=45)
            plot_idx += 1

        # Numeric Relationship: Scatter Plot
        if len(num_cols) >= 2:
            sns.scatterplot(x=self.df[num_cols[0]], y=self.df[num_cols[1]],
                            hue=self.df[cat_cols[0]] if cat_cols else None,
                            ax=axes[plot_idx], palette="deep")
            axes[plot_idx].set_title(f"{num_cols[0]} vs {num_cols[1]}")
            plot_idx += 1

        # Remove unused panels if dataset has fewer than 4 suitable charts
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])

        # Adjust layout
        fig.suptitle("Multi-Panel Data Visualization Dashboard", fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# =============================
# Example Usage
# =============================
if __name__ == "__main__":
    # Example dataset
    df = sns.load_dataset("tips")  # Built-in Seaborn dataset

    dashboard = MultiPanelDashboard(df)
    dashboard.plot_dashboard()
