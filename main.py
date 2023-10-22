import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

data = pd.read_csv('creditcard.csv')

legit = data[data.Class == 0]
fraud = data[data.Class == 1]

legit_sample = legit.sample(n=len(fraud), random_state=2)
balanced_data = pd.concat([legit_sample, fraud], axis=0)

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2, stratify=y)

model = LogisticRegression(solver='lbfgs', max_iter=3000)
model.fit(X_train, y_train)

train_accuracy = accuracy_score(model.predict(X_train), y_train)
test_accuracy = accuracy_score(model.predict(X_test), y_test)

# web App

st.title('Credit Card Fraud Detection')
input_df = st.text_input('Enter All 30 features of the transaction')
df_split = input_df.split(',')

submit = st.button('Submit')

if submit:
    features = np.asarray(df_split, dtype=np.float64)
    prediction = model.predict(features.reshape(1, -1))

    if prediction[0] == 0:
        st.write('Legitimate Transaction')
    else:
        st.write('Fraud Transaction')
