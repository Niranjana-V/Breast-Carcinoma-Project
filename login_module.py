import streamlit as st

def login():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("hospital_logo.jpg", width=100)
    with col2:
        st.markdown("<h2 style='color:#2E86C1; padding-top: 15px;'>🔐 Secure Login - MediScan Diagnostics</h2>", unsafe_allow_html=True)

    username = st.text_input("👤 Username", placeholder="Enter username")
    password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid credentials.")

def logout():
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()