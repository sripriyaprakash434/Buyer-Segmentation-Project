import streamlit as st
import pandas as pd

st.title("Buyer Segmentation Project")

clients = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

st.subheader("Clients Dataset")
st.dataframe(clients.head())

st.subheader("Properties Dataset")
st.dataframe(properties.head())
