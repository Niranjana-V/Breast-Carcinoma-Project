from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Save as CSV
df.to_csv("breast_cancer_data.csv", index=False)
print("Dataset saved as breast_cancer_data.csv")