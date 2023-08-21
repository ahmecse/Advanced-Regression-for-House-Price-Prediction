#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained Ridge model
filename = 'ridge_model.joblib'
loaded_ridge_model = joblib.load(filename)

# Streamlit web application
st.title('House Price Prediction App')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    object_columns = df.select_dtypes(include='object').columns.tolist()
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()

    # Check if all numerical columns exist in the DataFrame
    missing_columns = set(numerical_columns) - set(df.columns)
    if missing_columns:
        st.error(f"Missing columns in uploaded CSV: {', '.join(missing_columns)}")
    else:
        # Perform preprocessing for numerical columns
        skewness = df[numerical_columns].skew()
        skewed_columns = skewness[(skewness > 1) | (skewness < -1)]
        for feature in skewed_columns:
            df[feature] = np.log1p(df[feature])

        # Perform one-hot encoding for object columns
        df = pd.get_dummies(df, columns=object_columns, drop_first=True)

        # Predict using the loaded Ridge model
        prediction = loaded_ridge_model.predict(df)
        st.subheader('Predicted Price:')
        st.write(prediction)


# In[ ]:




