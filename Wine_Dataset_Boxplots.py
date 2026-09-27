import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

# Boxplots for all numerical attributes
plt.figure(figsize=(15, 10))
df.boxplot(rot=90)
plt.title("Boxplots of All Numerical Attributes in Wine Dataset")
plt.ylabel("Values")
plt.tight_layout()
plt.show()