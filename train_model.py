from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
import pandas as pd
import joblib

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Feature selection - keep top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Save the selected feature names
selected_features = selector.get_support(indices=True)
selected_names = X.columns[selected_features]
joblib.dump(selected_names.tolist(), "selected_features.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "xgboost_model_kbest.pkl")

print("✅ Model and selected features saved successfully.")