import streamlit as st
import pickle
import string
from preprocessing import text_preprocessing


hide_st_style = """
                <style>
                footer {visibility: hidden;}
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                #stException {visibility: hidden;}
                </style>
                """

st.markdown(hide_st_style, unsafe_allow_html=True)




TFIDF = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title("SMS spam classifier")
input_sms = st.text_input("Enter the sms")
if st.button("Predict"):
    # preprocess
    preprocess_sms = text_preprocessing(input_sms)


    # vectorize 
    vectorize_sms = TFIDF.transform([preprocess_sms])


    # predict
    result = model.predict(vectorize_sms)[0]

    # display output
    if result == 1:
        st.warning("This is a spam sms")
    else:
        st.balloons()
        st.success('This sms is not spam')
else:
    st.warning("Input box is empty")






  


st.text('Developed by Somnath Paul')




