# app.py  (merged XGBoost + Subclass Image predictor with Grad-CAM)
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import io
import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from login_module import login, logout
from feature_units import FEATURE_UNITS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage
import cv2
import os
import glob
from PIL import Image

# === New imports for subclass image predictor ===
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Layer
from tensorflow.keras.models import Model
import h5py

# ---------------- Load CSS ----------------
def apply_custom_css():
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

apply_custom_css()

# ---------------- Session Setup ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

logout()

# ---------------- Logo and Title ----------------
col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists("hospital_logo.jpg"):
        st.image("hospital_logo.jpg", width=90)
with col2:
    st.markdown("<h2 class='header-title'> Breast Carcinoma Prediction</h1>", unsafe_allow_html=True)

# ---------------- Email Function ----------------
def send_email(recipient_email, subject, body, fig=None):
    try:
        sender_email = "yourmail@gmail.com"
        sender_password = "your app password"  # keep secure in production!

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            msg.attach(MIMEImage(buf.read(), name="explanation.png"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        return True
    except Exception as e:
        st.error(f"❌ Email Error: {str(e)}")
        return False

# ---------------- Load Tabular Model(s) ----------------
model = joblib.load("xgboost_model_kbest.pkl")
selected_features = joblib.load("selected_features.pkl")
explainer = shap.Explainer(model)

# ---------- Subclass image predictor settings ----------
DEFAULT_CLASS_NAMES = [
    "Adenosis",
    "Fibroadenoma",
    "Phyllodes Tumor",
    "Tubular Adenoma",
    "Ductal Carcinoma",
    "Lobular Carcinoma",
    "Mucinous Carcinoma",
    "Papillary Carcinoma"
]
DEFAULT_IMG_SIZE = (300, 300)
tf.keras.backend.set_floatx("float32")

# ---------- Utilities for subclass model loading (safe) ----------
def find_h5_model():
    files = glob.glob("*.h5")
    if not files:
        raise FileNotFoundError("No .h5 file found in the current directory. Put your BreakHis .h5 here.")
    # prefer breakhis-looking filename
    files_sorted = sorted(files, key=lambda s: ("breakhis" not in s.lower(), s))
    return files_sorted[0]

class CastLayer(Layer):
    def _init_(self, dtype=None, **kwargs):
        super()._init_(**kwargs)
        try:
            self._target_dtype = tf.as_dtype(dtype) if dtype is not None else None
        except Exception:
            self._target_dtype = None

    def call(self, inputs, **kwargs):
        target = self._target_dtype if self._target_dtype is not None else tf.float32
        return tf.cast(inputs, target)

    def get_config(self):
        cfg = super().get_config()
        if self._target_dtype is not None:
            cfg.update({"dtype": self._target_dtype.name})
        return cfg

def build_efficientnetb3(num_classes, input_shape=(DEFAULT_IMG_SIZE[0], DEFAULT_IMG_SIZE[1], 3)):
    base = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation="softmax", dtype="float32")(x)
    return Model(inputs=base.input, outputs=out)

@st.cache_resource
def load_subclass_model(num_classes=len(DEFAULT_CLASS_NAMES)):
    """Try direct load; if fails, rebuild EfficientNetB3 and load weights by_name (skip_mismatch)."""
    model_file = find_h5_model()
    custom_objects = {"Cast": CastLayer}
    try:
        try:
            m = load_model(model_file, compile=False, custom_objects=custom_objects, safe_mode=False)
        except TypeError:
            m = load_model(model_file, compile=False, custom_objects=custom_objects)
        return m
    except Exception:
        m = build_efficientnetb3(num_classes)
        # attempt to load weights by name; skip mismatched layers
        m.load_weights(model_file, by_name=True, skip_mismatch=True)
        return m

# Cache-load subclass model (so manual/csv flows unaffected)
try:
    subclass_model = load_subclass_model()
except Exception as e:
    # don't crash the entire app; we'll handle missing model during image flow
    subclass_model = None
    st.warning(f"Could not load subclass model automatically: {e}")

# ---------- Grad-CAM helper ----------
def find_last_conv_layer(model):
    # try common name
    for name in ("top_conv", "block7a_project_conv", "top_conv"):
        try:
            _ = model.get_layer(name)
            return name
        except Exception:
            pass
    # fallback: find last layer with 'conv' in class name
    for layer in reversed(model.layers):
        if hasattr(layer, "output_shape"):
            cls_name = layer._class.name_.lower()
            if "conv" in cls_name:
                return layer.name
    raise ValueError("Could not find a convolutional layer in the model.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4, cmap="jet"):
    # pil_img: PIL RGB
    hmap = cv2.resize(heatmap, pil_img.size)  # resize to original
    # normalize to 0-255
    hmap_uint8 = np.uint8(255 * hmap)
    # apply colormap
    colored = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    # convert original to array
    img_arr = np.array(pil_img).astype(np.uint8)
    # blend
    blended = np.uint8(img_arr * (1 - alpha) + colored * alpha)
    return blended

# ---------------- Input Option ----------------
st.sidebar.header("⚙ Input Mode")
input_method = st.sidebar.radio("Choose:", ["Manual Entry", "Upload CSV File", "Image Upload"])

# ---------------- Manual Entry ----------------
if input_method == "Manual Entry":
    st.subheader("📝 Patient Entry Form")
    patient_name = st.text_input("👩‍⚕ Patient Name")

    col1, col2 = st.columns(2)
    input_data = []
    for i, feature in enumerate(selected_features):
        with (col1 if i % 2 == 0 else col2):
            unit = FEATURE_UNITS.get(feature, "")
            label = f"{feature} ({unit})" if unit else feature
            val = st.number_input(label, step=0.01)
            input_data.append(val)

    if st.button("🎯 Predict"):
        from ensemble_predict import predict_ensemble
        prediction, prob = predict_ensemble(input_data)
        result = "Benign" if prediction == 1 else "Malignant"
        st.session_state.manual_result = result
        st.session_state.manual_input_data = input_data

        shap_values = explainer([np.array(input_data)])
        st.session_state.manual_shap = shap_values

        color = "green" if result == "Benign" else "red"
        st.markdown(f"<h4 style='color:{color}'>🩺 Prediction: {result}</h4>", unsafe_allow_html=True)

        st.subheader("📊 SHAP Explanation")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        st.session_state.manual_fig = fig

        result_df = pd.DataFrame([input_data], columns=selected_features)
        result_df["Prediction"] = [result]
        shap_df = pd.DataFrame([shap_values[0].values], columns=selected_features)
        shap_df["Base Value"] = shap_values[0].base_values
        shap_df["Prediction Value"] = shap_values[0].values.sum() + shap_values[0].base_values
        final_df = pd.concat([result_df, shap_df], axis=1)
        st.download_button("📥 Download Result", final_df.to_csv(index=False), file_name="manual_result.csv", key="manual_download")

    if "manual_result" in st.session_state and "manual_fig" in st.session_state:
        st.subheader("📧 Email Report")
        patient_email = st.text_input("📨 Enter email address")
        if st.button("Send Email Report"):
            name_line = f"Patient Name: {patient_name}\n" if patient_name else ""
            email_sent = send_email(
                recipient_email=patient_email,
                subject="Mammary Carcinoma Prediction Report",
                body=f"{name_line}Prediction: {st.session_state.manual_result}\n\nSHAP explanation attached.",
                fig=st.session_state.manual_fig
            )
            if email_sent:
                st.success("✅ Email sent successfully.")
            else:
                st.error("❌ Failed to send email.")

# ---------------- CSV Upload ----------------
elif input_method == "Upload CSV File":
    st.subheader("📁 Upload Patient CSV")
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()

        if 'Email' not in df_raw.columns:
            st.error("❌ 'Email' column missing.")
            st.stop()

        try:
            df = df_raw[selected_features]
        except KeyError:
            st.error("❌ Columns mismatch with selected features.")
            st.stop()

        emails = df_raw['Email']
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            st.error("❌ No valid rows after cleaning.")
            st.stop()

        st.dataframe(df)
        
        if st.button("🎯 Predict from CSV"):
            from ensemble_predict import predict_ensemble
            predictions = [predict_ensemble(row)[0] for row in df.values]
            df_result = df.copy()
            df_result["Prediction"] = ["Benign" if p == 1 else "Malignant" for p in predictions]
    
            st.session_state.df_result = df_result
            st.session_state.emails = emails
            st.session_state.shap_values = shap.Explainer(model, df)(df)
            st.session_state.csv_uploaded = True

        if st.session_state.get("csv_uploaded", False):
            st.success("✅ Prediction completed!")
            st.dataframe(st.session_state.df_result)
            st.download_button("📥 Download CSV Results", st.session_state.df_result.to_csv(index=False), file_name="csv_predictions.csv")

            st.subheader("🔍 SHAP View for Row")
            selected_row = st.selectbox("Select row index", list(st.session_state.df_result.index))
            st.write(f"🧾 Prediction: {st.session_state.df_result['Prediction'].iloc[selected_row]}")
            fig, ax = plt.subplots()
            shap.plots.waterfall(st.session_state.shap_values[selected_row], max_display=10, show=False)
            st.pyplot(fig)

            if st.button("📧 Send All Emails"):
                success_count = 0
                for i in df.index:
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(st.session_state.shap_values[i], max_display=10, show=False)
                    sent = send_email(
                        recipient_email=emails[i],
                        subject="Mammary Carcinoma Prediction Report",
                        body=f"Prediction: {st.session_state.df_result['Prediction'].iloc[i]}\n\nSHAP explanation attached.",
                        fig=fig
                    )
                    if sent:
                        success_count += 1
                st.success(f"✅ Emails sent to {success_count}/{len(df)} patients.")

# ---------------- CNN Subclass Image Upload (REPLACED old binary image code) ----------------
elif input_method == "Image Upload":
    st.subheader("🖼 Upload Histopathology Image — Subclass Prediction")

    uploaded_img = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_img is None:
        st.info("Upload a histopathology image")
    else:
        # Show original uploaded image
        pil_img = Image.open(uploaded_img).convert("RGB")
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

        # Ensure subclass model loaded
        if subclass_model is None:
            st.error("❌ Subclass model not loaded. Make sure the BreakHis .h5 file is present in the app folder.")
            st.stop()

        # Preprocess image same as training
        arr = pil_img.resize(DEFAULT_IMG_SIZE, Image.LANCZOS)
        arr = np.asarray(arr).astype("float32")
        arr = preprocess_input(arr)
        input_batch = np.expand_dims(arr, axis=0)

        # Predict
        preds = subclass_model.predict(input_batch)[0]
        if not np.isclose(preds.sum(), 1.0, atol=1e-3):
            preds = tf.nn.softmax(preds).numpy()

        top_idx = int(np.argmax(preds))
        top_label = DEFAULT_CLASS_NAMES[top_idx]
        top_prob = float(preds[top_idx])

        st.subheader(f"✅ Prediction: {top_label}")
        st.write(f"*Confidence:* {top_prob:.4f}")

        # Top-3
        top3_idx = np.argsort(preds)[-3:][::-1]
        st.markdown("### Top-3 predictions")
        for i in top3_idx:
            st.write(f"- {DEFAULT_CLASS_NAMES[i]}: {preds[i]:.4f}")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(DEFAULT_CLASS_NAMES))
        ax.barh(y_pos, preds, color="tab:blue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(DEFAULT_CLASS_NAMES)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)
        ax.set_title("Class probabilities")
        st.pyplot(fig, use_container_width=True)

        # Grad-CAM option: compute and show heatmap
        if st.button("Show Grad-CAM Heatmap"):
            try:
                last_conv = find_last_conv_layer(subclass_model)
                heatmap = make_gradcam_heatmap(input_batch, subclass_model, last_conv_layer_name=last_conv, pred_index=top_idx)
                # overlay
                blended = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.4)
                # Display
                st.image(blended, caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.error(f"Grad-CAM failed: {e}")

        # Allow download of subclass prediction
        result_df = pd.DataFrame({
            "Prediction": [top_label],
            "Confidence": [top_prob]
        })
        st.download_button(
            label="📥 Download Subclass Prediction (CSV)",
            data=result_df.to_csv(index=False),
            file_name="subclass_prediction.csv",
            mime="text/csv"
        )