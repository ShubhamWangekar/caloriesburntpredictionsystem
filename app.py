# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# rfr = pickle.load(open('rfr.pkl','rb'))
# x_train = pd.read_csv('X_train.csv')
#
# def pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp):
#     features = np.array([[Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp]])
#     prediction = rfr.predict(features).reshape(1,-1)
#     return prediction[0]
#
#
# # web app
# # Gender Age Height Weight Duration Heart_Rate Body_Temp
# st.title("Calories Burn Prediction")
#
# Gender = st.selectbox('Gender', x_train['Gender'] )
# Age = st.selectbox('Age', x_train['Age'])
# Height = st.selectbox('Height', x_train['Height'])
# Weight = st.selectbox('Weight', x_train['Weight'])
# Duration = st.selectbox('Exercise Duration (minutes)', x_train['Duration'])
# Heart_rate = st.selectbox('Heart Rate (bpm)', x_train['Heart_Rate'])
# Body_temp = st.selectbox('Body Temperature', x_train['Body_Temp'])
#
# result = pred(Gender,Age,Height,Weight,Duration,Heart_rate,Body_temp)
#
# if st.button('predict'):
#     if result:
#         st.write("Total calories burn  :",result)
# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
#
# # Title and Description
# st.title("Calories Burned Prediction")
# st.write("Welcome to our calories burned prediction tool. Enter your activity details below to estimate the calories burned.")
#
# # Input Form
# st.sidebar.header("Input Parameters")
#
# activity_type = st.sidebar.selectbox("Select Activity Type", ["Running", "Walking", "Cycling"])
#
# if activity_type == "Running":
#     speed = st.sidebar.slider("Speed (km/h)", 5.0, 15.0, 8.0)
#     distance = st.sidebar.slider("Distance (km)", 1.0, 20.0, 5.0)
#     time = distance / speed
#
# elif activity_type == "Walking":
#     speed = st.sidebar.slider("Speed (km/h)", 3.0, 6.0, 5.0)
#     distance = st.sidebar.slider("Distance (km)", 1.0, 10.0, 3.0)
#     time = distance / speed
#
# else:  # Cycling
#     speed = st.sidebar.slider("Speed (km/h)", 10.0, 30.0, 20.0)
#     distance = st.sidebar.slider("Distance (km)", 5.0, 50.0, 15.0)
#     time = distance / speed
#
# # Calculate Calories Burned
# # Formula: Calories burned = MET * weight in kg * time in hours
# # MET values: Running: 7, Walking: 3.5, Cycling: 4
# # Assuming weight = 70 kg
#
# MET = {"Running": 7, "Walking": 3.5, "Cycling": 4}
# weight = 70  # kg
# calories_burned = MET[activity_type] * weight * time
#
# # Display Result
# st.subheader("Estimated Calories Burned")
# st.write(f"Estimated calories burned {activity_type.lower()} {distance} km at {speed} km/h: {calories_burned:.2f} kcal")
#



#
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# # Load the trained model
# rfr = pickle.load(open('rfr.pkl', 'rb'))
#
# # Load the training data
# x_train = pd.read_csv('X_train.csv')
#
# # Prediction function
# def pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
#     features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
#     prediction = rfr.predict(features)
#     return prediction[0]
#
# # Web app
# st.title("Calories Burn Prediction")
#
# Gender = st.selectbox('Gender', x_train['Gender'].unique())
# Age = st.selectbox('Age', x_train['Age'].unique())
# Height = st.selectbox('Height', x_train['Height'].unique())
# Weight = st.selectbox('Weight', x_train['Weight'].unique())
# Duration = st.selectbox('Exercise Duration (minutes)', x_train['Duration'].unique())
# Heart_rate = st.selectbox('Heart Rate (bpm)', x_train['Heart_Rate'].unique())
# Body_temp = st.selectbox('Body Temperature (°C)', x_train['Body_Temp'].unique())
#
# if st.button('Predict'):
#     result = pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)
#     st.write("Total calories burned:", result)



import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
rfr = pickle.load(open('rfr.pkl', 'rb'))

# Prediction function
def pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp):
    features = np.array([[Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp]])
    prediction = rfr.predict(features)
    return prediction[0]

# Web app
st.title("Calories Burn Prediction")

# Gender selection remains a select box
Gender = st.selectbox('Gender', ['male', 'female'])
Gender = 1 if Gender == 'male' else 0

# Input fields for other features
Age = st.number_input('Age', min_value=0, max_value=100, value=25)
Height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
Weight = st.number_input('Weight (kg)', min_value=10, max_value=200, value=70)
Duration = st.number_input('Exercise Duration (minutes)', min_value=0, max_value=300, value=30)
Heart_rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200, value=70)
Body_temp = st.number_input('Body Temperature (°C)', min_value=30.0, max_value=45.0, value=36.5)

if st.button('Predict'):
    result = pred(Gender, Age, Height, Weight, Duration, Heart_rate, Body_temp)
    st.write("Total calories burned:", result)


