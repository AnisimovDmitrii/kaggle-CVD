import streamlit as st
import pickle




def model_load():
    with open('models/model.pcl', 'rb') as fid:
        return  pickle.load(fid)

features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol',
            'gluc', 'smoke', 'alco', 'active', 'age_year', 'bmi']


age_year = st.slider('age', 0, 120, key='age_year')
height = st.slider('height', 60, 200, key='height')
weight = st.slider('weight', 35, 150, key='weight')
ap_hi = st.slider('ap_hi', 60, 220, key='ap_hi')
ap_lo = st.slider('ap_lo', 40, 150, key='ap_lo')

gender = st.sidebar.selectbox('gender', 1, 2, key='gender')
cholesterol = st.sidebar.selectbox('cholesterol', 1, 2, 3, key='cholesterol')
gluc = st.sidebar.selectbox('gluc', 1, 2, key='gluc')

smoke = st.checkbox('Do you smoke?', key='smoke')
alco = st.checkbox('Do you drink alcohol?', key='alco')
active = st.checkbox('Do you make sport?', key='active')


age = age_year * 365
bmi = weight / (height / 100) ** 2


model = model_load(1)

pred = model.predict_proba[features][:,1]

st.write(pred)