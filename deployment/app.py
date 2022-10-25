from codecs import latin_1_decode
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

preprocess = pickle.load(open('preprocess_churn1.pkl', 'rb'))
model = tf.keras.models.load_model('model_churn1.h5')

st.header('Aplikasi Mengecek Status Customer/Pelangan PT INIHOME')


tenure = st.number_input('masa sewa Customer/Pelangan')
MonthlyCharges = st.number_input('Biaya Bulanan Customer/Pelangan')
TotalCharges = st.number_input('Total Biaya Sewa Customer/Pelangan')

st.write('Silakan Menjawab Beberapa pertanyaan berikut:')
SeniorCitizen = st.selectbox('Apakah anda orangtua atau anak muda? jika orangtua pilih 1, jika anak muda pilih 0', [0, 1])
PaymentMethod = st.selectbox('Metode pembayaran apa yang anda gunakan ketika berlangganan?', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
PaperlessBilling = st.selectbox('Apakah bukti pembayaran anda berupa kertas atau tidak?', ['Yes', 'No'])
Contract = st.selectbox('Berapa lama anda berlangganan?', ['Month-to-month', 'One year', 'Two year'])
Partner = st.selectbox('Apakah anda memiliki pasangan atau tidak?', ['Yes', 'No'])
Dependents = st.selectbox('Apakah anda memiliki tangungan (seperti anak atau orangtua) atau tidak?', ['Yes', 'No'])
TechSupport = st.selectbox('Apakah anda jaringan INIHOME dapat membantu anda dalam mengunakan internet/browser?', ['Yes', 'No', 'No internet service'])
OnlineSecurity = st.selectbox('Apakah anda merasa aman dalam mengunakan jaringan INIHOME?', ['Yes', 'No', 'No internet service'])
InternetService = st.selectbox('Apakah pelayanan internet yang anda gunakan?', ['DSL', 'Fiber optic', 'No internet service'])


if st.button('submit'):
    
    feature_num = ['tenure', 'MonthlyCharges', 'TotalCharges']
    feature_nonum = ['SeniorCitizen']
    feature_cat = ['PaymentMethod', 'PaperlessBilling', 'Contract', 'Partner', 'Dependents', 'TechSupport', 'OnlineSecurity', 'InternetService']

    num_df = pd.DataFrame([[tenure, MonthlyCharges, TotalCharges]], columns=feature_num)
    no_df  = pd.DataFrame([[SeniorCitizen]], columns=feature_nonum)
    cat_df = pd.DataFrame([[PaymentMethod, PaperlessBilling, Contract, Partner, Dependents, TechSupport, OnlineSecurity, InternetService]], columns=feature_cat)

    X = pd.concat([num_df, no_df, cat_df], axis=1)

    trans_X = pd.DataFrame(preprocess.transform(X))


  
    pred = model.predict(trans_X)
    


    if pred[0][0] < 0.5:
        st.text('Congratulation Anda termasuk Customer No Churn, \n Terimakasih telah setia memakai layanan internet kami, \n berikan kritik dan saran anda melalui no wa CS INIHOME: 081234567890')
    else:
        st.text('Congratulation Anda termasuk Customer Churn, \n Terimakasih telah memakai layanan internet kami, \n mohon maaf apabila ada kesalahan dalam jaringan kami , \n jika berkenan, silakan berikan kritik dan saran anda ke no wa CS INIHOME: 081234567890')

  
