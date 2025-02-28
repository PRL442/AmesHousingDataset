#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:56:10 2025

@author: emilypinson
"""

# Load libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import openpyxl
import xlrd

# Define the file name manuallly
file_name = "AmesHousing.xlsx" # Set your filename here
print(f"Using file: {file_name}")

# Read the dataset from the local file in the GitHub repository
df = pd.read_excel('AmesHousing.xlsx')

# Load the Excel file
data = pd.read_excel(file_name, sheet_name=0) # Load first sheet

# Clean column names
data.columns = [col.split('(')[0].strip() for col in data.columns]
data.rename(columns={'SalePrice': 'Sale Price'}, inplace=True)

# Assuming no missing values, split the data into features and target
X = data.drop(columns=['Sale Price'])
y = data['Sale Price']

# Train a Multiple Regression Model 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Create the streamlit web-based app

# Title of the app
st.title('Ames Housing Dataset Predictions')

# Sidebar for user inputs
st.sidebar.header('Input Parameters')

def user_input_features():
    Lot_Area = st.sidebar.slider('Lot Area', 1300, 215245, 2000)
    Overall_Qual = st.sidebar.slider('Overall Quality', 1, 10, 5)
    Overall_Cond = st.sidebar.slider('Overall Condition', 1, 10, 5)
    Year_Built = st.sidebar.slider('Year Built', 1872, 2010, 1999)
    Total_Bsmt_SF = st.sidebar.slider('Total Basement SF', 0, 6110, 2500)
    
    data = {
        'Lot Area': Lot_Area,
        'Overall Quality': Overall_Qual,
        'Overall Condition': Overall_Cond,
        'Year Built': Year_Built,
        'Total Basement SF': Total_Bsmt_SF,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Parameters')
st.write(input_df)

# Predict the housing price
prediction = model.predict(input_df)

# Display the prediction
st.subheader('Sale Price Predictor (dollars)')
st.write(prediction[0])