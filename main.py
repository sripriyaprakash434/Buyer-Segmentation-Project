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


st.subheader("Data Types")

st.write(data.dtypes)

st.subheader("Clustering Dataset")

cluster_data = data[[
    "age",
    "sale_price",
    "floor_area_sqft",
    "satisfaction_score"
]]

st.dataframe(cluster_data.head())

st.write("Missing Values")
st.write(cluster_data.isnull().sum())

cluster_data["age"] = cluster_data["age"].fillna(
    cluster_data["age"].median()
)

st.write("Missing Values After Filling")
st.write(cluster_data.isnull().sum())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_data)

st.subheader("Scaled Data")

st.write("Scaled Data Shape:", scaled_data.shape)

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

st.subheader("Elbow Method")

fig, ax = plt.subplots()

ax.plot(range(1, 11), wcss, marker="o")

ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method")

st.pyplot(fig)

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

st.subheader("Elbow Method")

fig, ax = plt.subplots()

ax.plot(range(1, 11), wcss, marker="o")

ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")

st.pyplot(fig)

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

st.subheader("Elbow Method")

fig, ax = plt.subplots()

ax.plot(range(1, 11), wcss, marker="o")

ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")

st.pyplot(fig)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

data["cluster"] = kmeans.fit_predict(scaled_data)

st.subheader("Cluster Counts")
st.write(data["cluster"].value_counts())

st.subheader("Buyer Segmentation")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    data["age"],
    data["sale_price"],
    c=data["cluster"]
)

ax.set_xlabel("Age")
ax.set_ylabel("Sale Price")
ax.set_title("Buyer Segmentation")

st.pyplot(fig)

st.pyplot(fig)

st.download_button(
    label="Download Buyer Segments CSV",
    data=data.to_csv(index=False),
    file_name="buyer_segments.csv",
    mime="text/csv"
)


st.subheader("Project Presentation")

with open(
    "Data_ Analyst_ Buyer Segmentation Project Using K-Means Clustering.pdf",
    "rb"
) as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(
    label="📄 Download Project Presentation",
    data=PDFbyte,
    file_name="Buyer_Segmentation_Presentation.pdf",
    mime="application/pdf"
)
