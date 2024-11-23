#!/usr/bin/env python
# coding: utf-8

# # Streamlit application

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import r2_score


# Load preprocessed data
file_path = r"Horizontalus ruozas-Eksperimetu suvestine (version 2).xlsx"
df1 = pd.read_excel(file_path, sheet_name='Sheet3')
df2 = pd.read_excel(file_path, sheet_name='New')
df1 = df1.drop(index=df1.index[0]).reset_index(drop=True)
df2 = df2.drop(index=df1.index[0]).reset_index(drop=True)

# Preprocessing
df1['Heat_transfer_coefficient'] = df1[['First_heat_Coef',
                                        'Second_heat_Coef',
                                        'Third_heat_Coef',
                                        'Fourth_heat_Coef',
                                        'Fifth_heat_Coef',
                                        'Sixth_heat_Coef',
                                        'Seventh_heat_Coef',
                                        'Eighth_heat_Coef']].mean(axis=1)

X_features_1 = df1[['Mixture tin, oC', 'Mass Fraction', 'Dewpoint',
                    'Mixture  (air+vapour) flow rate, kg/h', 'Cooling water flow rate, l/h', 'Cooling H2O tin, oC']]
X_features_2 = df2[['SPECIFIC HEAT (kj/kg k)', 'VISCOSITY_MIX[μPa s]',
                    'THERMAL CONDUCTIVTY (W/m k)', "LATENT HEAT OF VAPOURIZATION [KJ/Kg]"]]
X_features_combined = pd.concat([X_features_1, X_features_2], axis=1)
y = df1['Heat_transfer_coefficient'].values

# Scale data
scaler_x = MinMaxScaler()
X = scaler_x.fit_transform(X_features_combined)
scaler_y = MinMaxScaler()
Y = scaler_y.fit_transform(y.reshape(-1, 1))

# Load trained model
model = load_model('best_model0.keras')  # Ensure you save the trained model with this name
# model = joblib.load('clf_gra_model.pkl')

# Streamlit App
st.title("Heat Transfer Coefficient Prediction")

# User selects experiment
st.sidebar.header("Experiment Selection")
experiment_index = st.sidebar.selectbox(
    "Choose an experiment:", options=range(len(X)), format_func=lambda x: f"Experiment {x + 1}")

# Predict and compare
if experiment_index is not None:
    selected_data = X[experiment_index].reshape(1, -1)
    predicted = model.predict(selected_data)
    actual = Y[experiment_index]

    # Inverse transform the predictions and actual values
    predicted_original = scaler_y.inverse_transform(predicted)[0][0]
    # Check the shape of the actual value first
    print(np.shape(actual))  # See what shape the variable actually has
    
    # If actual is a scalar, it will be reshaped as (1, 1)
    if np.ndim(actual) == 0:
        actual_reshaped = np.reshape([actual], (1, 1))  # Reshape to 2D if scalar
    else:
        # If actual is already an array, reshape it properly
        actual_reshaped = np.reshape(actual, (-1, 1))  # Reshape to 2D
    
    # Apply inverse_transform
    actual_original = scaler_y.inverse_transform(actual_reshaped)[0][0]


    # Calculate error
    error_percentage = np.abs(predicted_original - actual_original) / actual_original * 100

    # Display results
    st.subheader(f"Experiment {experiment_index + 1} Results")
    st.write(f"**Predicted Heat Transfer Coefficient:** {predicted_original:.2f}")
    st.write(f"**Actual Heat Transfer Coefficient:** {actual_original:.2f}")
    st.write(f"**Error Percentage:** {error_percentage:.2f}%")

    # Bar chart
    st.subheader("Comparison Chart")
    fig, ax = plt.subplots()
    ax.bar(["Predicted", "Actual"], [predicted_original, actual_original], color=['blue', 'orange'])
    ax.set_ylabel("Heat Transfer Coefficient")
    ax.set_title("Comparison of Predicted vs Actual")
    st.pyplot(fig)
  
    fig, ax = plt.subplots()
    yy = model.predict(X)
    yy = scaler_y.inverse_transform(yy)
    plt.figure(figsize=(10,4))
    ax.scatter(np.arange(1, len(yy)+1, 1), df1['Heat_transfer_coefficient'])
    # plt.scatter(np.arange(1, len(yy)+1, 1), yy)
    ax.plot(np.arange(1, len(yy)+1, 1), yy, ls='--', color='r')
    r2 = r2_score(df1['Heat_transfer_coefficient'], yy)
    st.write(f"Test Set R² Score: {r2}")
    st.pyplot(fig) 

# Checkbox for custom input
if st.checkbox("Enter Custom Data for Prediction"):
    st.subheader("Enter Input Features")

    # Input form
    form = st.form(key="custom_input_form")
    col1, col2 = form.columns(2)

    with col1:
        mixture_temp = form.number_input("Mixture Temperature (°C)", value=25.0)
        mass_fraction = form.number_input("Mass Fraction", value=0.5)
        dewpoint = form.number_input("Dewpoint", value=10.0)
        mixture_flow_rate = form.number_input("Mixture Flow Rate (kg/h)", value=100.0)
        cooling_flow_rate = form.number_input("Cooling Water Flow Rate (l/h)", value=50.0)
        cooling_temp = form.number_input("Cooling Water Inlet Temp (°C)", value=15.0)

    with col2:
        specific_heat = form.number_input("Specific Heat (kJ/kg K)", value=1.0, format="%.9f")
        viscosity = form.number_input("Viscosity (μPa s)", value=10.00, format="%.9f")
        thermal_conductivity = form.number_input("Thermal Conductivity (W/m K)", value=0.1, format="%.9f")
        latent_heat = form.number_input("Latent Heat of Vaporization (kJ/kg)", value=2200.0)

    submit_button = form.form_submit_button(label="Predict")

    if submit_button:
        # Prepare custom input
        custom_input = np.array([[mixture_temp, mass_fraction, dewpoint, mixture_flow_rate,
                                  cooling_flow_rate, cooling_temp, specific_heat,
                                  viscosity, thermal_conductivity, latent_heat]])
        custom_input_scaled = scaler_x.transform(custom_input)

        # Predict
        custom_prediction = model.predict(custom_input_scaled)
        custom_prediction_original = scaler_y.inverse_transform(custom_prediction)[0][0]

        st.write(f"**Predicted Heat Transfer Coefficient:** {custom_prediction_original:.2f}")


# In[ ]:




