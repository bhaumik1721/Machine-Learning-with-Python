import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

def load_lottie_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        animation_data = json.load(f)
    return animation_data

def preprocess_batch_file(df):
    type_mapping = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    df['type'] = df['type'].map(type_mapping)
    return df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]

# Load the model
model_path = 'DeploymentFinal.joblib'
model = joblib.load(model_path)

# Function to predict fraud
def predict_fraud(data):
    prediction = model.predict(data)
    return prediction

success_animation_path = "LegitmatePayment.json"
error_animation_path = "FraudPayment.json"
success_animation = load_lottie_file(success_animation_path)
error_animation = load_lottie_file(error_animation_path)

# Load images for logo'
logo_image_url = 'Logo.png'

# Streamlit app
st.set_page_config(page_title="Online Payment Fraud Detection", page_icon=logo_image_url, layout="wide")

# Adding logo
st.image(logo_image_url, width=200)

st.title('Online Payment Fraud Detection')

# Adding tabs for navigation
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header('Single Transaction Prediction')

    st.subheader('Input Features')

    type_options = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    type_encoded = st.selectbox('Type', range(len(type_options)), format_func=lambda x: type_options[x])
    amount = st.number_input('Amount', min_value=0.0, format="%.2f")
    oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0, format="%.2f")
    newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0, format="%.2f")
   

    data = np.array([[type_encoded, amount, oldbalanceOrg, newbalanceOrig]])

    if st.button('Predict'):
        result = predict_fraud(data)
        if result == 1:
            st.error('Warning: This transaction is predicted to be fraudulent.')
            st_lottie(error_animation, height=300, key="error")
        else:
            st.success('This transaction is predicted to be legitimate.')
            st_lottie(success_animation, height=300, key="success")

with tab2:
    st.header('Batch Prediction: Upload a CSV File')

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        processed_data = preprocess_batch_file(data)

        predictions = model.predict(processed_data)

        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].map({0: 'Not Fraud', 1: 'Fraud'})

        st.write(data)

        # Visualize the predictions using Plotly
        prediction_counts = data['Prediction'].value_counts().reset_index()
        prediction_counts.columns = ['Prediction', 'count']
        
        fig = px.bar(prediction_counts, x='Prediction', y='count',
                     labels={'Prediction': 'Prediction', 'count': 'Count'},
                     title='Fraud vs Not Fraud Predictions',
                     color='Prediction',
                     color_discrete_map={'Not Fraud': 'green', 'Fraud': 'red'})
        st.plotly_chart(fig)

        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)

        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

st.markdown('<div style="text-align: center; font-size: 90%; color: #888;">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
