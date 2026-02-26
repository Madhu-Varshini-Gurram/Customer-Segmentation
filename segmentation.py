import streamlit as st
import pandas as pd
import numpy as np
import joblib


kmeans = joblib.load('customer_segmentation_kmeans_model.pkl')
scaler = joblib.load('customer_segmentation_scaler.pkl')

st.title("Customer Segmentation App")
st.write("Upload customer details for segmentation.")

age=st.number_input("Age", min_value=18, max_value=100, value=30)
income=st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spend=st.number_input("Total Spend(Sum of Purchases)", min_value=0,max_value=5000, value=1000)
num_web_purchases=st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases=st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_web_visits=st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency=st.number_input("Recency (Days since last purchase)", min_value=0, max_value=365, value=30)


input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_Spent': [total_spend],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [num_web_visits],
    'Recency': [recency]

})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: {cluster}")
    