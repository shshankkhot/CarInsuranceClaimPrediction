import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Car Insurance Claim Prediction",
    page_icon="👋",
)


st.title("Car Insurance Claim Prediction")
st.markdown("Predict whether the policyholder will file a claim in the next 6 months or not.")
image = Image.open('image/CAR.jpg')
st.image(image)


#st.UserInfoProxy

#@st.cache
def wrangle(filepath):
    df = pd.read_csv(filepath).set_index("policy_id")
    
    ## droping high cardinality features
    df = df.drop("area_cluster", axis = 1)
    
    return df

df = wrangle("data/train/train.csv")
df_test = wrangle("data/test/test.csv")

def main():
    st.markdown("Predict whether the policyholder will file a claim in the next 6 months or not.")
    image = Image.open('image\CAR.jpg')
    st.image(image)    

def About_Data():
    st.markdown("# About Data 🎈")
    st.sidebar.markdown("#About Data🎈")
    
def eda():
    st.markdown("# EDA")
    st.sidebar.markdown("# Exploratory Data Analysis (EDA)")    

#def page2(df):
#    st.markdown("# Page 2 ❄️")
#    st.sidebar.markdown("# Page 2 ❄️")

def model_bld():
    st.markdown("Building Model")
    st.sidebar.markdown("Model Building")

#main_page()
