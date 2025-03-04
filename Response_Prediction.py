import streamlit as st
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

# Load the dataframe
df = pd.read_excel('marketing_campaign.xlsx')

# Load the saved LightGBM model and LabelEncoders
model = joblib.load('lightgbm_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to encode categorical inputs
def encode_inputs(input_data, encoders):
    for col, encoder in encoders.items():
        if col in input_data:
            try:
                # Try to transform the data
                input_data[col] = encoder.transform([input_data[col]])[0]
            except ValueError:
                # If a ValueError occurs (i.e., unseen label), use a fallback strategy
                input_data[col] = encoder.transform([encoder.classes_[0]])[0]  # Fallback to the first class
    return input_data

# Feature Engineering
def feature_engineering(df):
    df["Education_level"] = "Low"
    if education in ["Graduation", "PhD", "Master"]:
        df["Education_level"] = "high"
    elif education in ["Basic"]:
        df["Education_level"] = "Middle"

    df["Living_Status"] = "Living with Others"
    if marital_status in ["Alone", "Absurd", "YOLO"]:
        df["Living_Status"] = "Living Alone"

    df["Age"] = 2022 - df["Year_Birth"]
    input_data['Total_Campaigns_Accepted'] = (input_data['AcceptedCmp1'] +
                                               input_data['AcceptedCmp2'] +
                                               input_data['AcceptedCmp3'] +
                                               input_data['AcceptedCmp4'] +
                                               input_data['AcceptedCmp5'])
    input_data['Average_Spend'] = (input_data['MntWines'] + input_data['MntFruits'] + input_data['MntMeatProducts'] +
                                   input_data['MntFishProducts'] + input_data['MntSweetProducts'] +
                                   input_data['MntGoldProds']) / input_data['NumDealsPurchases']
    df['Is_Parent'] = int(df['Kidhome'] + df['Teenhome'] > 0)
    df['total_spending'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + df['MntSweetProducts'] + df['MntGoldProds']
    df['avg_web_visits'] = df['NumWebVisitsMonth'] / 12
    df['online_purchase_ratio'] = df['NumWebPurchases'] / (df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases'])
    if 'Year_Birth' in input_data:
        del input_data['Year_Birth']
    return df

# Streamlit UI
st.title("Customer Response Prediction")
st.write("Enter customer details to predict if they will respond positively to a campaign.")

# Input fields
with st.form(key='customer_form'):
    col1, col2 = st.columns(2)
    with col1:
        year_birth = st.number_input('Year of Birth', min_value=1900, max_value=2022, value=1970)
        education = st.selectbox("Education", df['Education'].unique())
        marital_status = st.selectbox("Marital Status", df['Marital_Status'].unique())
        income = st.number_input("Income", min_value=0, max_value=100000, step=1000)
        kidhome = st.slider("Number of Kids", 0, 2, 0)
        teenhome = st.slider("Number of Teenagers", 0, 2, 0)
        recency = st.slider("Recency (days since last purchase)", 0, 100, 0)
        
    with col2:
        num_deals_purchases = st.number_input('Num Deals Purchases', min_value=0, value=1)
        num_web_purchases = st.number_input('Num Web Purchases', min_value=0, value=1)
        num_catalog_purchases = st.number_input('Num Catalog Purchases', min_value=0, value=1)
        num_store_purchases = st.number_input('Num Store Purchases', min_value=0, value=1)
        num_web_visits = st.number_input('Num Web Visits Month', min_value=0, value=1)
        complain = st.selectbox('Complain', [0, 1])

    # Accepted Campaigns
    st.subheader("Campaign Interactions")
    col1, col2, col3 = st.columns(3)
    with col1:
        accepted_cmp1 = st.selectbox("Accepted Campaign 1", [0, 1])
    with col2:
        accepted_cmp2 = st.selectbox("Accepted Campaign 2", [0, 1])
    with col3:
        accepted_cmp3 = st.selectbox("Accepted Campaign 3", [0, 1])

    col1, col2, col3 = st.columns(3)
    with col1:
        accepted_cmp4 = st.selectbox("Accepted Campaign 4", [0, 1])
    with col2:
        accepted_cmp5 = st.selectbox("Accepted Campaign 5", [0, 1])

    # Amount spent
    st.subheader("Amount Spent")
    col1, col2, col3 = st.columns(3)
    with col1:
        mnt_wines = st.number_input("Amount Spent on Wine", min_value=0, step=10)
    with col2:
        mnt_fruits = st.number_input("Amount Spent on Fruits", min_value=0, step=10)
    with col3:
        mnt_meat = st.number_input("Amount Spent on Meat", min_value=0, step=10)

    col1, col2, col3 = st.columns(3)
    with col1:
        mnt_fish = st.number_input("Amount Spent on Fish", min_value=0, step=10)
    with col2:
        mnt_sweets = st.number_input("Amount Spent on Sweets", min_value=0, step=10)
    with col3:
        mnt_gold = st.number_input("Amount Spent on Gold", min_value=0, step=10)

    # Prepare input data dictionary
    input_data = {
        "Year_Birth": year_birth,
        "Education": education,
        "Marital_Status": marital_status,
        "Income": income,
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Recency": recency,
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweets,
        "MntGoldProds": mnt_gold,
        'NumDealsPurchases': num_deals_purchases,
        'NumWebPurchases': num_web_purchases,
        'NumCatalogPurchases': num_catalog_purchases,
        'NumStorePurchases': num_store_purchases,
        'NumWebVisitsMonth': num_web_visits,
        "AcceptedCmp1": accepted_cmp1,
        "AcceptedCmp2": accepted_cmp2,
        "AcceptedCmp3": accepted_cmp3,
        "AcceptedCmp4": accepted_cmp4,
        "AcceptedCmp5": accepted_cmp5,
        'Complain': complain
    }

    # Submit button
    submit_button = st.form_submit_button(label="Predict Response")

    if submit_button:
        # Apply feature engineering
        input_data = feature_engineering(input_data)

        # Encode categorical features
        input_data = encode_inputs(input_data, label_encoders)

        # Convert input_data to DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        response_text = "Positive Response" if prediction == 1 else "No Response"
        st.success(f"Predicted Response: {response_text}")
