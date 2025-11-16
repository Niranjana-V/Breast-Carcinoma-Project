import pandas as pd
import joblib

# Load models
xgb_model = joblib.load("xgboost_model_kbest.pkl")
lgb_model = joblib.load("lgb_model_kbest.pkl")
selected_features = joblib.load("selected_features.pkl")

# Weighted prediction function
def predict_ensemble(input_data, w_xgb=0.6, w_lgb=0.4):
    """
    input_data: list or 1D array of feature values in the order of selected_features
    returns: predicted class (0 or 1) and probability array
    """
    input_df = pd.DataFrame([input_data], columns=selected_features)
    xgb_pred_proba = xgb_model.predict_proba(input_df)
    lgb_pred_proba = lgb_model.predict_proba(input_df)

    final_pred_proba = w_xgb * xgb_pred_proba + w_lgb * lgb_pred_proba
    final_pred = final_pred_proba.argmax(axis=1)[0]

    return final_pred, final_pred_proba[0]