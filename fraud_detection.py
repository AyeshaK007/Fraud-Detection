import streamlit as st
import pandas as pd 
import joblib 

# Load the pre-trained machine learning pipeline
model = joblib.load("fraud_detection_pipeline.pkl")

# Set up the title and header layout for the web app
st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and use the predict button.")
st.divider()

# Input field for Categorical feature: Transaction Type
transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
)

# Input fields for Numerical features
amount = st.number_input(
    "Amount", 
    min_value=0.0, 
    value=1000.0
)

old_balance_original = st.number_input(
    "Old Balance (Sender)", 
    min_value=0.0, 
    value=10000.0
)

new_balance_original = st.number_input(
    "New Balance (Sender)", 
    min_value=0.0, 
    value=9000.0
)

old_balance_destination = st.number_input(
    "Old Balance (Receiver)", 
    min_value=0.0, 
    value=0.0
)

new_balance_destination = st.number_input(
    "New Balance (Receiver)", 
    min_value=0.0, 
    value=0.0
)

# Execution trigger when the Predict button is clicked
if st.button("Predict"):
    
    # Structure the user inputs exactly into a DataFrame matching the model's expected feature names
    input_data = pd.DataFrame([{
    "type": transaction_type,
    "amount": amount,
    "oldbalanceOrg": old_balance_original,  # Sender ka purana balance
    "newbalanceOrig": new_balance_original,  # Sender ka naya balance
    "oldbalanceDest": old_balance_destination, # Receiver ka purana balance
    "newbalanceDest": new_balance_destination  # Receiver ka naya balance
}])
    
    # Generate the prediction (0 for Legitimate, 1 for Fraudulent)
    prediction = model.predict(input_data)[0]
    
    st.subheader(f"Prediction Result: {int(prediction)}")
    
    # Conditional UI display based on the model's flag
    if prediction == 1:
        st.error("Warning: This transaction matches known high-risk anomalies and could be fraudulent!")
    else:
        st.success("Clear: This transaction looks normal and is likely safe.")