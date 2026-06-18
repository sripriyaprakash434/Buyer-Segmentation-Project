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

clients["loan_applied"].value_counts().plot()

st.subheader("Country Distribution")

fig, ax = plt.subplots()

clients["country"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Country")
ax.set_ylabel("Number of Clients")

st.pyplot(fig)

    
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)
