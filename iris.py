import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
df = pd.concat([X, y], axis=1)

# Create a big plot with many subplots
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))

# Plot scatterplots for all pairs of features
for i, feature1 in enumerate(df.columns[:-1]):
    for j, feature2 in enumerate(df.columns[:-1]):
        if i == j:
            axs[i, j].hist(df[feature1])
        else:
            axs[i, j].scatter(df[feature2], df[feature1], c=df['species'], cmap='viridis')
        if j == 0:
            axs[i, j].set_ylabel(feature1)
        if i == len(df.columns[:-1]) - 1:
            axs[i, j].set_xlabel(feature2)

# Adjust subplot spacing
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.4, wspace=0.4)

# Save the plot to a PNG file
plt.savefig('iris_plot.png')

# Show the plot
plt.show()
