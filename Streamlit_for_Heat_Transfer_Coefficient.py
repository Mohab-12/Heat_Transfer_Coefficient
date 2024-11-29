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
DelT1 = df1['Temperature decrease of the mixture_1'] - df1['Temperature increase of the cooling water_1']
DelT2 = df1['Temperature decrease of the mixture_9'] - df1['Temperature increase of the cooling water_9']

DelT_LM = (DelT1-DelT2)/np.log(DelT1/DelT2)
df1['DelT_LM'] = DelT_LM 
df1['Heat_transfer_coefficient'] = df1['Heat water'] / (df1['DelT_LM']*0.0206*8)

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
# model = load_model('best_model0.keras')  # Ensure you save the trained model with this name
KNN = joblib.load('knn_model.pkl')
Rf = joblib.load('Rf_model.pkl')
# clf_gra = joblib.load('clf_gra_model.pkl')
# Function to calculate thermophysical properties
def thermo_phy(T_g, MF, m_g, M_h2o=18.015, M_g=28.96):
    m_v = MF * m_g  # vapour flow rate
    y_h2o = (m_v / M_h2o) / ((m_v / M_h2o) + ((m_g - m_v) / M_g))  # vapour mole fraction
    y_air = 1 - y_h2o

    # Specific heat
    cp_Air = np.interp(T_g, [0, 25, 50, 100, 150, 200], [1.006, 1.007, 1.01, 1.02, 1.03, 1.04])
    cp_water = np.interp(T_g, [0, 100, 200, 300, 400], [1.8, 2.0, 2.2, 2.4, 2.6])
    M_m = y_air * M_g + y_h2o * M_h2o
    c_pg = ((M_g / M_m) * y_air * cp_Air + (M_h2o / M_m) * y_h2o * cp_water)

    # Viscosity
    u_air = np.interp(T_g, [0, 25, 50, 100, 150, 200], [0.017, 0.018, 0.019, 0.02, 0.021, 0.022])
    u_water = np.interp(T_g, [0, 100, 200, 300], [0.001, 0.0008, 0.0006, 0.0004])
    viscosity = y_air * u_air + y_h2o * u_water

    # Thermal conductivity
    k_air = np.interp(T_g, [0, 100, 200, 300], [0.02, 0.03, 0.04, 0.05])
    k_water = np.interp(T_g, [0, 100, 200, 300], [0.6, 0.65, 0.7, 0.75])
    thermal_conductivity = y_air * k_air + y_h2o * k_water

    # Latent heat of vaporization
    latent_heat = 2257 - 2.5 * T_g

    return c_pg, viscosity, thermal_conductivity, latent_heat



# Streamlit App
st.title("Heat Transfer Coefficient Prediction")

# Checkbox for custom input
# if st.checkbox("Enter Custom Data for Prediction"):
st.subheader("Enter Input Features")

  # Input form
  # form = st.form(key="custom_input_form")
  # col1, col2 = form.columns(2)

  # with col1:
mixture_temp = st.number_input("Mixture Temperature (°C)", value=25.0)
mass_fraction = st.number_input("Mass Fraction", value=0.5)
dewpoint = st.number_input("Dewpoint", value=10.0)
mixture_flow_rate = st.number_input("Mixture Flow Rate (kg/h)", value=100.0)
cooling_flow_rate = st.number_input("Cooling Water Flow Rate (l/h)", value=50.0)
cooling_temp = st.number_input("Cooling Water Inlet Temp (°C)", value=15.0)
know_properties = st.radio("Do you know the thermophysical properties?", ("Yes", "No"))
if know_properties == "Yes":
    specific_heat = st.number_input("Specific Heat (kJ/kg K)", value=1.0)
    viscosity = st.number_input("Viscosity (μPa s)", value=10.0)
    thermal_conductivity = st.number_input("Thermal Conductivity (W/m K)", value=0.1)
    latent_heat = st.number_input("Latent Heat of Vaporization (kJ/kg)", value=2200.0)
else:
    c_pg, viscosity, thermal_conductivity, latent_heat = thermo_phy(mixture_temp-10, mass_fraction, mixture_flow_rate)
    st.write(f"Calculated Specific Heat: {c_pg:.4f} kJ/kg K")
    st.write(f"Calculated Viscosity: {viscosity:.4f} μPa s")
    st.write(f"Calculated Thermal Conductivity: {thermal_conductivity:.4f} W/m K")
    st.write(f"Calculated Latent Heat of Vaporization: {latent_heat:.4f} kJ/kg")

    # Auto-populate fields
    specific_heat = c_pg


submit_button = st.button(label="Predict")

if submit_button:
    try:
        # Prepare custom input
        custom_input = np.array([[mixture_temp, mass_fraction, dewpoint, mixture_flow_rate,
                                  cooling_flow_rate, cooling_temp, specific_heat,
                                  viscosity, thermal_conductivity, latent_heat]])

        # Scale the input data
        custom_input_scaled = scaler_x.transform(custom_input)

        # Predictions using the models
        custom_prediction_KNN = KNN.predict(custom_input_scaled)
        custom_prediction_original_KNN = scaler_y.inverse_transform(custom_prediction_KNN.reshape(-1, 1))[0][0]

        custom_prediction_Rf = Rf.predict(custom_input_scaled)
        custom_prediction_original_Rf = scaler_y.inverse_transform(custom_prediction_Rf.reshape(-1, 1))[0][0]

        # custom_prediction_clf_gra = clf_gra.predict(custom_input_scaled)
        # custom_prediction_original_clf_gra = scaler_y.inverse_transform(custom_prediction_clf_gra.reshape(-1, 1))[0][0]

        # Display the results
        st.write(f"**Predicted Heat Transfer Coefficient (KNN):** {custom_prediction_original_KNN:.2f}")
        st.write(f"**Predicted Heat Transfer Coefficient (Random Forest):** {custom_prediction_original_Rf:.2f}")
        # st.write(f"**Predicted Heat Transfer Coefficient (Gradient Boosting):** {custom_prediction_original_clf_gra:.2f}")

        # Visualization
        fig, ax = plt.subplots()
        ax.bar(["KNN", "Random Forest"],
               [custom_prediction_original_KNN, custom_prediction_original_Rf],
               color=['blue', 'green', 'orange'])
        ax.set_ylabel("Heat Transfer Coefficient")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# In[ ]:




