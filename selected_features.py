from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)

selected_features = X.columns[selector.get_support()]
print("✅ Top 10 selected features:\n", selected_features.tolist())