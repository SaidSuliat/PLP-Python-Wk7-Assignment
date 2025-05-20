# Analyzing Data with Pandas and Visualizing Results with Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame
df.rename(columns={"target": "species"}, inplace=True)
df["species"] = df["species"].apply(lambda x: iris.target_names[x])

# Exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Basic statistics
print(df.describe())

# Group by species
print(df.groupby("species").mean())

# Visualizations
sns.set(style="whitegrid")

# Line Chart
df.groupby("species")["sepal length (cm)"].cumsum().plot()
plt.title("Cumulative Sepal Length per Species")
plt.xlabel("Index")
plt.ylabel("Cumulative Sepal Length (cm)")
plt.legend(["Setosa", "Versicolor", "Virginica"])
plt.tight_layout()
plt.savefig("line_chart.png")
plt.close()

# Bar Chart
sns.barplot(x="species", y="petal length (cm)", data=df)
plt.title("Average Petal Length by Species")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.close()

# Histogram
sns.histplot(df["sepal width (cm)"], bins=15, kde=True)
plt.title("Distribution of Sepal Width")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

# Scatter Plot
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs. Petal Length")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()
