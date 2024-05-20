import streamlit as st
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble

data = pd.read_csv('housing_ml.csv')

st.title('**Housing Price Prediction App**')

st.sidebar.header('User Input Parameters')

def user_input_features():

    min_value = 150
    max_value = 3000
    step = 50 

    Consituency = st.sidebar.selectbox('Constituency', ['harare north', 'harare west', 'harare south', 'harare east'])
    Beds = st.sidebar.slider('Beds', 1, 10, 2)
    Baths = st.sidebar.slider('Baths', 1, 5, 2)
    Area_sqm = st.sidebar.slider('Area_Sqm', min_value, max_value, 150, step = step)
    data = {'Constituency': Consituency,
            'Beds': Beds, 
            'Baths': Baths,
            'Area_Sqm': Area_sqm}
    
    features = pd.DataFrame(data, index = [0])
    return features


df = user_input_features()
region = df.iloc[0, 0]

if region == 'harare south' or region == 'harare west':
    #st.write('harare west or south building still underway')
    housing_con = data[data['Constituency'] == region]
    st.write(df)

    Y = housing_con['Price'].reset_index(drop = True)
    X = housing_con[['Beds', 'Baths', 'Area_Sqm']].reset_index(drop = True)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 2)

    #clf = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'squared_error')
    #clf.fit(x_train, y_train)
    #prediction_prob = clf.score(x_test, y_test)

    #prediction = clf.predict(df.drop(columns = 'Constituency', axis = 1))
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    prediction_prob = reg.score(x_train, y_train)
    prediction = reg.predict(df.drop(columns = 'Constituency', axis = 1))

    #st.subheader('Prediction')
    st.subheader('A house with these specs wil cost US$:')
    st.write(prediction)

    st.subheader('Prediction Accuracy')
    st.write(prediction_prob)

elif region == 'harare east' or region == 'harare north':
    housing_con = data[data['Constituency'] == region]
    st.write(df)

    Y = housing_con['Price'].reset_index(drop = True)
    X = housing_con[['Beds', 'Baths', 'Area_Sqm']].reset_index(drop = True)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 2)

    clf = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'squared_error')
    clf.fit(x_train, y_train)
    prediction_prob = clf.score(x_test, y_test)

    prediction = clf.predict(df.drop(columns = 'Constituency', axis = 1))

    #st.subheader('Prediction')
    st.subheader('A house with these specs wil cost US$:')
    st.write(prediction)

    st.subheader('Prediction Accuracy')
    st.write(prediction_prob)




#st.subheader('User Input Parameters')
#st.write(df)

#region = df.iloc[0, 0]
#st.write(region)

#housing_con = data[data['Constituency'] == region]
#st.write(housing_constituency.head())

#Y = housing_con['Price'].reset_index(drop = True)
#X = housing_con[['Beds', 'Baths', 'Area_Sqm']].reset_index(drop = True)

#x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.10, random_state = 2)

#clf = ensemble.GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'squared_error')
#clf.fit(x_train, y_train)
#prediction_prob = clf.score(x_test, y_test)

#prediction = clf.predict(df.drop(columns = 'Constituency', axis = 1))

#st.subheader('Prediction')
#st.subheader('A house with these specs wil cost US$:')
#st.write(prediction)

#st.subheader('Prediction Accuracy')
#st.write(prediction_prob)