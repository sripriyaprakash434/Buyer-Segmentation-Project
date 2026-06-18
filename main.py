import streamlit as st
import pandas as pd

st.title("Buyer Segmentation Project")

clients = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

st.subheader("Clients Dataset")
st.dataframe(clients.head())

st.subheader("Properties Dataset")
st.dataframe(properties.head())


import matplotlib.pyplot as plt

st.subheader("Client Type Distribution")

fig, ax = plt.subplots()
clients["client_type"].value_counts().plot(kind="bar", ax=ax)

st.pyplot(fig)

st.subheader("Loan Applied Distribution")

fig, ax = plt.subplots()

clients["loan_applied"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)

st.subheader("Country Distribution")

fig, ax = plt.subplots()

clients["country"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Country")
ax.set_ylabel("Number of Clients")

st.pyplot(fig)


st.subheader("Clients Dataset Information")

st.write("Clients Shape:", clients.shape)
st.write("Properties Shape:", properties.shape)

from datetime import datetime

clients["date_of_birth"] = pd.to_datetime(
    clients["date_of_birth"],
    errors="coerce"
)

clients["age"] = (
    datetime.now().year
    - clients["date_of_birth"].dt.year
)

st.subheader("Age Calculation")

st.dataframe(
    clients[["date_of_birth", "age"]].head()
)

st.write("Client Columns")
st.write(clients.columns)

st.subheader("Merged Dataset")

data = pd.merge(
    clients,
    properties,
    left_on="client_id",
    right_on="client_ref",
    how="inner"
)

st.write("Merged Dataset Shape:", data.shape)

st.dataframe(data.head())

st.write("Property Columns")
st.write(properties.columns)


st.subheader("Sale Price Conversion")

data["sale_price"] = (
    data["sale_price"]
    .replace(r"[$,]", "", regex=True)
    .astype(float)
)

st.write(data["sale_price"].head())
