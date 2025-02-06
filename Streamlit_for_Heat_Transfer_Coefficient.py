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
import math
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

X_features_1 = df1[['Mixture tin, oC', 'Mass Fraction', 
                    'Cooling water flow rate, l/h', 'Cooling H2O tin, oC']]
# X_features_2 = df2[['SPECIFIC HEAT (kj/kg k)', 'VISCOSITY_MIX[μPa s]',
#                     'THERMAL CONDUCTIVTY (W/m k)', "LATENT HEAT OF VAPOURIZATION [KJ/Kg]"]]
X_features_combined = X_features_1
y = df1['Heat_transfer_coefficient'].values

# Scale data
scaler_x = MinMaxScaler()
X = scaler_x.fit_transform(X_features_combined)
scaler_y = MinMaxScaler()
Y = scaler_y.fit_transform(y.reshape(-1, 1))

# Load trained model
# model = load_model('Fake_data_model.keras')  # Ensure you save the trained model with this name
KNN = joblib.load('knn_fake_model.pkl')
Rf = joblib.load('Rf_fake_model.pkl')
svr = joblib.load('svr__fake_model.pkl')

# Function to calculate thermophysical properties
def thermo_phy(T_g, MF, m_g, M_h2o=18.015, M_g=28.96):
  m_v = MF*m_g       #vapour flow rate
  print(f"The mass flow rate of vapour is {m_v:.2f}")
  y_h2o = (m_v/M_h2o)/((m_v/M_h2o)+((m_g-m_v)/M_g)) #Vapour mole fraction
  print(f"The mole fraction of vapour is {y_h2o:.2f}")
  y_air = 1 - y_h2o
  print(f"The mole fraction of air is {y_air:.2f}")
# Specific heat
  values1 = [0, 6.9, 15.6, 26.9, 46.9, 66.9, 86.9, 107, 127, 227, 327, 427, 527, 627]
  values2 = [1.006, 1.006, 1.006, 1.006, 1.007, 1.009, 1.01, 1.012, 1.014, 1.03, 1.051, 1.075, 1.099, 1.121]
  point = float(T_g)
  cp_Air = np.interp(point, values1, values2)
  values3 = [-23, 2, 27, 52, 77, 102, 127, 177, 227, 277, 327, 377, 427, 477, 527, 577, 627, 677, 727]
  values4 = [1.855, 1.859, 1.864, 1.871, 1.88, 1.89, 1.901, 1.926, 1.954, 1.984, 2.015, 2.047, 2.08, 2.113, 2.147, 2.182, 2.217, 2.252, 2.288]
  point = float(T_g)
  cp_water = np.interp(point, values3, values4)
  M_m = y_air*M_g + y_h2o*M_h2o
  c_pg = ((M_g/M_m)*y_air*cp_Air + (M_h2o/M_m)*y_h2o*cp_water)
  print ("Specific heat of the air side {} Kj/Kg.K".format(np.round(c_pg,4)))
#Viscosity
  values1 = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200, 225, 300]
  values2 = [0.00001715, 0.0000174, 0.00001764, 0.00001789, 0.00001813, 0.00001837, 
                   0.0000186, 0.00001907, 0.00001953, 0.00001999, 0.00002088, 0.00002174, 
                   0.00002279, 0.0000238, 0.00002478, 0.00002573, 0.00002666, 0.00002928]
  u_air = np.interp(point, values1, values2) 
  values3 = [ 17.51, 24.1, 28.98, 32.9, 36.18, 39.02, 41.53, 43.79, 45.83, 60.09, 69.13, 75.89, 81.35, 85.95, 89.96,
                 93.51, 96.71, 99.63, 102.32, 104.81, 107.13, 109.32, 111.37, 111.37, 113.32, 115.17, 116.93,
                 118.62, 120.23, 123.27, 126.09, 128.73, 131.2, 133.54, 138.87, 143.63, 147.92, 151.85,
                 155.47, 158.84, 161.99, 164.96, 167.76, 170.42, 172.94, 175.36, 177.67, 179.88, 184.06,
                 187.96, 191.6, 195.04, 198.28, 201.37, 204.3, 207.11, 209.79, 212.37, 214.85, 217.24,
                 219.55, 221.78, 223.94, 226.03, 228.06, 230.04, 231.96, 233.84]
  values4 = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.000011, 0.000011,
                      0.000011, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000012, 0.000013,
                      0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013, 0.000013,
                      0.000013, 0.000013, 0.000013, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014, 0.000014,
                      0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015, 0.000015,
                      0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016, 0.000016,
                      0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017, 0.000017]
  u_water = np.interp(point, values3, values4) 
  Q_av = (math.sqrt(2) / 4) * (1 + (M_g / M_h2o)) ** -0.5 * ((1 + math.sqrt(u_air / u_water)) * (M_h2o / M_g) ** 0.25) ** 2
  Q_va = (math.sqrt(2) / 4 )*( 1 + (M_h2o / M_g)) ** -0.5 * ((1 + math.sqrt(u_water / u_air)) * (M_g / M_h2o) ** 0.25) ** 2
  viscosity = ((y_air*u_air)/((y_air) + (y_h2o*Q_av))) + ((y_h2o*u_water)/((y_h2o) + (y_air*Q_va)))
  print("Dynamic viscosity of the mixture {} μPa s".format(np.round(viscosity,10)))
#Thermal conductivity
  values = [-190, -150, -100, -75, -50, -25, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150,
            175, 200, 225, 300, 412, 500, 600, 700, 800, 900, 1000, 1100]
  numbers = [7.82, 11.69, 16.2, 18.34, 20.41, 22.41, 23.2, 23.59, 23.97, 24.36, 24.74, 25.12, 25.5, 25.87, 26.24,
             26.62, 27.35, 28.08, 28.8, 30.23, 31.62, 33.33, 35, 36.64, 38.25, 39.83, 44.41, 50.92, 55.79, 61.14,
             66.32, 71.35, 76.26, 81.08, 85.83]
  k_g_air = np.interp(T_g,values,numbers)/1000
  numbers1 = [0.01, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 360]
  numbers2 = [0.0171, 0.0173, 0.0176, 0.0179, 0.0182, 0.0186, 0.0189, 0.0192, 0.0196, 0.02, 0.0204, 0.0208, 0.0212, 0.0216, 0.0221, 0.0225, 0.023, 0.0235, 0.024, 0.0246, 0.0251, 0.0262, 0.0275, 0.0288, 0.0301, 0.0316, 0.0331, 0.0347, 0.0364, 0.0382, 0.0401, 0.0442, 0.0487, 0.054, 0.0605, 0.0695, 0.0836, 0.11, 0.178]
  k_g_vapour = np.interp(T_g,numbers1,numbers2)
  thermal_conductivity = ((1-y_h2o)*k_g_air)/((1-y_h2o)+Q_av*y_h2o) + (y_h2o*k_g_vapour)/(y_h2o+(1-y_h2o)*Q_va)  # (W/m k) thermal conductivity of air
  print("Thermal conductivity of air {} W/m k".format(np.round(thermal_conductivity,4)))
#Latent heat
  latent_heat = -0.0021 * (T_g**2) - 2.2115 *(T_g) + 2499
  print('Latent heat of vapourization {} Kj/kg'.format(np.round(latent_heat,4)))
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
#dewpoint = st.number_input("Dewpoint", value=10.0)
#mixture_flow_rate = st.number_input("Mixture Flow Rate (kg/h)", value=100.0)
cooling_flow_rate = st.number_input("Cooling Water Flow Rate (l/h)", value=50.0)
cooling_temp = st.number_input("Cooling Water Inlet Temp (°C)", value=15.0)
# know_properties = st.radio("Do you know the thermophysical properties?", ("Yes", "No"))
#if know_properties == "Yes":
    #specific_heat = st.number_input("Specific Heat (kJ/kg K)", value=1.0)
    #viscosity = st.number_input("Viscosity (μPa s)", value=10.0)
    #thermal_conductivity = st.number_input("Thermal Conductivity (W/m K)", value=0.1)
    #latent_heat = st.number_input("Latent Heat of Vaporization (kJ/kg)", value=2200.0)
#else:
 #   c_pg, viscosity, thermal_conductivity, latent_heat = thermo_phy(mixture_temp-10, mass_fraction/100, mixture_flow_rate)
  #  st.write(f"Calculated Specific Heat: {c_pg:.9f} kJ/kg K")
  #  st.write(f"Calculated Viscosity: {viscosity:.9f} μPa s")
  #  st.write(f"Calculated Thermal Conductivity: {thermal_conductivity:.9f} W/m K")
  #  st.write(f"Calculated Latent Heat of Vaporization: {latent_heat:.9f} kJ/kg")

    # Auto-populate fields
 #   specific_heat = c_pg


submit_button = st.button(label="Predict")

if submit_button:
    try:
        # Prepare custom input
        custom_input = np.array([[mixture_temp, mass_fraction, 
                                  cooling_flow_rate, cooling_temp]])

        # Scale the input data
        custom_input_scaled = scaler_x.transform(custom_input)

        # Predictions using the models
        custom_prediction_KNN = KNN.predict(custom_input_scaled)
        custom_prediction_original_KNN = scaler_y.inverse_transform(custom_prediction_KNN.reshape(-1, 1))[0][0]

        custom_prediction_Rf = Rf.predict(custom_input_scaled)
        custom_prediction_original_Rf = scaler_y.inverse_transform(custom_prediction_Rf.reshape(-1, 1))[0][0]

        custom_prediction_svr = svr.predict(custom_input_scaled)
        custom_prediction_original_svr = scaler_y.inverse_transform(custom_prediction_svr.reshape(-1, 1))[0][0]

        # Display the results
        st.write(f"**Predicted Heat Transfer Coefficient (KNN):** {custom_prediction_original_KNN:.2f}")
        st.write(f"**Predicted Heat Transfer Coefficient (Random Forest):** {custom_prediction_original_Rf:.2f}")
        # st.write(f"**Predicted Heat Transfer Coefficient (SVM):** {custom_prediction_original_svr:.2f}")

        # Visualization
        fig, ax = plt.subplots()
        ax.bar(["KNN", "Random Forest","SVR"],
               [custom_prediction_original_KNN, custom_prediction_original_Rf, custom_prediction_original_svr],
               color=['blue', 'green', 'orange'])
        ax.set_ylabel("Heat Transfer Coefficient")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# In[ ]:




