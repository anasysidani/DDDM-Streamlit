import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
df = pd.read_csv('BMW.csv')
df = df.drop(['car_model'], axis = 'columns')
df['Age']=2022-df['year']
df = df.drop(['year'], axis = 'columns')
df = df.drop(['tax'], axis = 'columns')
df=df.drop(['fuel_type','trans'],axis='columns')
st.set_page_config(layout="wide")


st.title('Used BMW Price Predictor')
st.write('---')
with st.sidebar:
    selected = option_menu(menu_title=None,
    options=["Home","Car Price"],
    icons=["house","gear"],
    menu_icon="cast",
    default_index=0,
    styles={
    "container": {"padding":"0!important"},"icon": {"color":"#blue"},
    "nav-link": {
    "font-size":"15px",
    "text-align":"left",
    "margin":"Opx","--hover-color":"#eee"},
    "nav-link-selected": {"background-color": "#454545"},
    },)
    
    if selected =="Home":
        
     st.image("data:image/jpeg;base64,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")

    with st.expander("Why Use Our App?"):
     st.write("""
         We Created this application so salespeople at our company know the stimated selling price of a used car based on the inputted specs  \nline
         By using this app, the salespeople are able to know how much to buy the car from the seller.  \nline
         This application takes the guess work out of the equation, salespeople will be able to know how much the car could be sold for, and buy for cheaper.   \nline
         Using this application will increase revenues inevitably \nline
           
     """)

    
#creating the sliders in sidebar
def user_input_features():
      mileage = st.sidebar.slider('mileage', float(df.mileage.min()),float(df.mileage.max()),value= float(df.mileage.mean()))
      mpg = st.sidebar.slider('mpg',float(df.mpg.min()), float(df.mpg.max()), float(df.mpg.mean()))
      engine_size = st.sidebar.slider('engine size', min_value=float(df.engine_size.min()), max_value=float(df.engine_size.max()), value=float(df.engine_size.mean()))
      Age = st.sidebar.slider('Age', min_value=float(df.Age.min()), max_value=float(df.Age.max()), value=float(df.Age.mean()))
      data = {'mileage': mileage,
            'mpg': mpg,     
            'engine_size': engine_size,
            'Age':Age}
      features = pd.DataFrame(data, index=[0])
      return features

input_df=user_input_features()

X = df.drop(['price'], axis='columns')
y=df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train.values, y_train.values)

st.header('Specified Input parameters')
st.write(input_df)
pred=decision_tree.predict(input_df)
st.header('Estimated Price')

for i in pred:
               st.write("Estimated selling Price: " , round(i ,0)  , "$")
               st.write("Buy Car for", round(i ,0)-2000,"$"," or less")

st.write('---')