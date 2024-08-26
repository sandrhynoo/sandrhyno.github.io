import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Fraud Prediction App", page_icon=":mag:", layout="wide")

st.write("""
# Credit Card Fraud Prediction App

This is a website application predicting credit card fraud!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        cardPresent = st.sidebar.selectbox('Card Present',('yes','no'))
        matchCVV = st.sidebar.selectbox('MatchCVV',('yes','no'))
        transactionType = st.sidebar.selectbox('Transaction Type',('PURCHASE','REVERSAL','ADDRESS_VERIFICATION'))
        posEntryMode = st.sidebar.selectbox('Pos Entry Mode', ('A','B','C','D','E'))
        transactionAmount = st.sidebar.number_input('Transaction Amount', 0.0,1825.25,111.33)
        data = {'cardPresent': cardPresent,
                'matchCVV': matchCVV,
                'transactionType': transactionType,
                'posEntryMode': posEntryMode,
                'transactionAmount': transactionAmount}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
transaction_raw = pd.read_csv('data.csv')
transaction = transaction_raw.drop(columns=['isFraud'])
df = pd.concat([input_df,transaction],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['cardPresent','transactionType','matchCVV','posEntryMode']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('transaction_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
transaction_isFraud = np.array(['no','yes'])
st.write(transaction_isFraud[prediction])


# st.subheader('Prediction Probability')
# st.write(prediction_proba)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
