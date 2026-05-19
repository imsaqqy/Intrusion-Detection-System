import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="IDS Detector", layout="wide")

# --- LOAD MODEL ARTIFACTS ---
@st.cache_resource
def load_model():
    if os.path.exists('hybrid_artifacts.joblib'):
        artifacts = joblib.load('hybrid_artifacts.joblib')
        return artifacts
    else:
        st.error("Model file 'hybrid_artifacts.joblib' not found. Please run the training script first.")
        return None

artifacts = load_model()

# --- HELPER FUNCTION: PREPROCESSING ---
def preprocess_input(df, expected_features):
    # One-hot encode the input
    df_encoded = pd.get_dummies(df)
    
    # Ensure all columns from training exist
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Remove extra columns and reorder to match model training
    df_final = df_encoded[expected_features]
    return df_final

# --- UI LAYOUT ---
st.title("🛡️ Cyber Threat Detection System")
st.markdown("""
This application uses a **XGBoost + LightGBM Model** to detect network intrusions 
based on the NSL-KDD dataset format.
""")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Network Traffic (CSV)", type=["csv"])

if uploaded_file is not None and artifacts:
    # 1. Read Data
    input_df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(input_df.head(10))

    # 2. Preprocess & Predict
    with st.spinner('Analyzing network traffic...'):
        model = artifacts['model']
        feature_cols = artifacts['features']
        le = artifacts['label_encoder']

        # Align features
        processed_data = preprocess_input(input_df, feature_cols)
        
        # Make predictions
        preds = model.predict(processed_data)
        attack_names = le.inverse_transform(preds)
        
        # Add results to dataframe
        input_df['Detection_Result'] = attack_names

    # 3. Display Results
    st.success("Analysis Complete!")
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Detection Summary")
        counts = input_df['Detection_Result'].value_counts()
        st.write(counts)
        
        # Highlight threats
        threats = counts.drop('normal', errors='ignore').sum()
        st.metric("Total Threats Detected", threats, delta=f"{threats} alerts", delta_color="inverse")

    with col2:
        st.subheader("Threat Distribution")
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Detailed Results")
    st.dataframe(input_df)

    # 4. Download Results
    csv = input_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='threat_detection_results.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload a CSV file in the sidebar to begin the analysis.")