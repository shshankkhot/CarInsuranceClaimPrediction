import pandas as pd
import streamlit as st
import numpy as np
import main

df = main.df
st.header("About Dataset")

st.markdown("The Dataset contains information on policyholders having the attributes like policy tenure, age of the car, age of the car owner, the population density of the city, make and model of the car, power, engine type, etc, and the target variable indicating whether the policyholder files a claim in the next 6 months or not.")

st.write(df.head())
#st.write(df.describe)
# View the summary of the dataset
#st.write(df.info(verbose=False))

st.markdown('''**Comments:**
                \n-> This info reflects that the dataset has no null values.
                \n-> There are 28 Categorical features and 16 numerical features.''')

col1,col2 = st.columns(2)
with col1:
    #categorical features
    categorical = df.select_dtypes(include =[np.object])
    #st.header("Categorical")
    st.write("Categorical Features in DataSet:",categorical.shape[1])
    st.write(categorical.columns)

with col2:
    #numerical features
    numerical= df.select_dtypes(include =[np.float64,np.int64])
    st.write("Numerical Features in DataSet:",numerical.shape[1])
    st.write(numerical.columns)

train_df = main.df
test_df = main.df_test

st.markdown("**Shape of the train and test dataset**")
st.write(train_df.shape, test_df.shape)