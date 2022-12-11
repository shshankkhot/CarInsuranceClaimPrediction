import streamlit as st
from PIL import Image
st.markdown('# Car Insurance Claim Prediction')
st.markdown("Predict whether the policyholder will file a claim in the next 6 months or not.")
image = Image.open('image\CAR.jpg')
st.image(image)

