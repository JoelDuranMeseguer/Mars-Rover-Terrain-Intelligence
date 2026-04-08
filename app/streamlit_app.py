"""Minimal Streamlit app scaffold for Mars Rover Terrain Intelligence."""

import streamlit as st


st.set_page_config(page_title="Mars Rover Terrain Intelligence", layout="centered")
st.title("🚀 Mars Rover Terrain Intelligence")
st.write("This is a starter Streamlit scaffold for terrain segmentation and navigation demos.")

uploaded = st.file_uploader("Upload a terrain image", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    st.success("Image uploaded. Inference integration comes next.")
