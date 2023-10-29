import pickle
import numpy as np
import streamlit as st

model = pickle.load(open('penyakitjantung.sav', 'rb'))
st.title('Prediksi Penyakit Jantung')

with st.form(key='heart_form'):
    age = st.number_input('Umur', min_value=0)
    sex = st.radio('Jenis Kelamin', options=['Laki-laki', 'Perempuan'])
    cp = st.selectbox('Tingkat Sakit Dada', options=[0, 1, 2, 3])
    trestbps = st.number_input('Tekanan Darah', min_value=0)
    chol = st.number_input('Kolestrol', min_value=0)
    fbs = st.radio('Gula Darah', options=['Kurang dari 120 mg/dl', 'Lebih dari 120 mg/dl'])
    restecg = st.selectbox('Hasil Elektrokadiografi', options=[0, 1, 2])
    thalach = st.number_input('Detak Jantung Maksimum', min_value=0)
    exang = st.radio('Induksi Angina', options=['Tidak', 'Ya'])
    oldpeak = st.number_input('ST Depression', min_value=0.0)
    slope = st.selectbox('Slope', options=[0, 1, 2])
    ca = st.number_input('Nilai CA', min_value=0)
    thal = st.selectbox('Nilai Thal', options=[0, 1, 2, 3])

    submitted = st.form_submit_button('Lakukan Prediksi Penyakit Jantung')

if submitted:
    
    sex = 1 if sex == 'Laki-laki' else 1
    fbs = 1 if fbs == 'Lebih dari 120 mg/dl' else 0
    exang = 1 if exang == 'Ya' else 0

    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    heart_prediction = model.predict(input_data)

    if heart_prediction[0] == 1:
        heart_diagnosis = 'Pasien Terkena Penyakit Jantung'
    else:
        heart_diagnosis = 'Pasien Tidak Terkena Penyakit Jantung'
    st.success(heart_diagnosis)
