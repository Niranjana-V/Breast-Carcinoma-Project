import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Load selected features from XGBoost training
selected_features = joblib.load("selected_features.pkl")
X_new = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

# Train LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
lgb_model.fit(X_train, y_train)

# Save model
joblib.dump(lgb_model, "lgb_model_kbest.pkl")
print("✅ LightGBM model saved successfully.")