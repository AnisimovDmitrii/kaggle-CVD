import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostClassifier

model = CatBoostClassifier()

model.load_model("models/model_cbc")


st.title('Check your probability cardiovascular disease')

col1, col2 = st.columns(2)

with col1:
    gender = st.radio('What\'s your gender?',
                              ('Female', 'Male'), 
                              key='gender')

with col2:
    active = st.radio('Do you make sport?',
                        ('yes' ,'no'), 
                        key='active')

with col1:
    smoke = st.radio('Do you smoke?',
                        ('yes' ,'no'), 
                        key='smoke')

with col2:
    alco = st.radio('Do you drink alcohol?',
                        ('yes' ,'no'), 
                        key='alco')


age_year = st.slider('What\'s your age', 0, 120, 40, key='age_year')
height = st.slider('What\'s your height', 60, 200, 175, key='height')
weight = st.slider('What\'s your weight', 35, 150, 75, key='weight')
ap_hi = st.slider('What\'s your systolic blood pressure', 60, 220, 120, key='ap_hi')
ap_lo = st.slider('What\'s your diastolic blood pressure', 40, 150, 80, key='ap_lo')

col2_1, col2_2 = st.columns(2)

with col2_1:
    cholesterol = st.radio('What\'s your cholesterol?',
                              (1, 2, 3), 
                              key='cholesterol')

with col2_2:
    gluc = st.radio('What\'s your gluc?',
                        (1, 2, 3), 
                        key='gluc')


if gender == 'Female':
    gender = 1
elif gender == 'Male':
    gemder = 2


if smoke == 'yes':
    smoke = 1
else:
    smoke = 0

if alco == 'yes':
    alco = 1
else:
    alco = 0   

if active == 'yes':
    active = 1
else:
    active = 0

age = age_year * 365
bmi = weight / (height / 100) ** 2


pred = model.predict_proba([age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_year, bmi])[1]


prob = str(round(pred * 100))
prob = prob + '%'

col3_1, col3_2 = st.columns(2)

with col3_1:
    st.subheader('Your CVD probabilities:')
with col3_2:
    st.subheader(prob)