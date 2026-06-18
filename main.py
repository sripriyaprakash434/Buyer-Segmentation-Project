import pandas as pd
import matplotlib.pyplot as plt

clients = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

print("Clients Dataset")
print(clients.head())

print("\nProperties Dataset")
print(properties.head())

print("Clients Shape:", clients.shape)
print("Properties Shape:", properties.shape)

print(clients.columns)
print(properties.columns)

print(clients.info())
print(properties.info())

print(clients.isnull().sum())
print(properties.isnull().sum())

print(clients.duplicated().sum())
print(properties.duplicated().sum())

clients = clients.drop_duplicates()
properties = properties.drop_duplicates()

# Client Type Distribution
clients["client_type"].value_counts().plot(kind="bar")

plt.title("Client Type Distribution")
plt.xlabel("Client Type")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Loan Applied Distribution
clients["loan_applied"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Loan Applied Distribution")
plt.ylabel("")
plt.show()

# Country Distribution
clients["country"].value_counts().plot(kind="bar")

plt.title("Country Distribution")
plt.xlabel("Country")
plt.ylabel("Number of Clients")

plt.show()

print(clients.columns)
print(properties.columns)

from datetime import datetime

clients["date_of_birth"] = pd.to_datetime(
    clients["date_of_birth"],
    errors="coerce"
)

clients["age"] = (
    datetime.now().year
    - clients["date_of_birth"].dt.year
)

print(clients[["date_of_birth", "age"]].head())
# Merge datasets

# Merge datasets

data = pd.merge(
    clients,
    properties,
    left_on="client_id",
    right_on="client_ref",
    how="inner"
)

print("\nMerged Dataset Shape:")
print(data.shape)

print("\nMerged Dataset Preview:")
print(data.head())

# Clean sale_price

data["sale_price"] = (
    data["sale_price"]
    .replace(r"[$,]", "", regex=True)
    .astype(float)
)

print("\nSale Price Converted:")
print(data["sale_price"].head())

# Clean sale_price

data["sale_price"] = (
    data["sale_price"]
    .replace(r"[$,]", "", regex=True)
    .astype(float)
)

print("\nSale Price Converted:")
print(data["sale_price"].head())

# Check data types

print("\nData Types:")
print(data.dtypes)

# Select features for clustering

cluster_data = data[[
    "age",
    "sale_price",
    "floor_area_sqft",
    "satisfaction_score"
]]

print("\nClustering Dataset:")
print(cluster_data.head())

print("\nMissing Values:")
print(cluster_data.isnull().sum())

# Fill missing ages with median

cluster_data["age"] = cluster_data["age"].fillna(
    cluster_data["age"].median()
)

print("\nMissing Values After Filling:")
print(cluster_data.isnull().sum())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_data)

print("\nScaled Data Shape:")
print(scaled_data.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(cluster_data)

print("\nScaled Data Shape:")
print(scaled_data.shape)

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

plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_data)

data["cluster"] = clusters

print("\nCluster Counts:")
print(data["cluster"].value_counts())


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    data["age"],
    data["sale_price"],
    c=data["cluster"]
)

plt.title("Buyer Segmentation")
plt.xlabel("Age")
plt.ylabel("Sale Price")

plt.show()


print("\nCluster Summary:")
print(
    data.groupby("cluster")[
        ["age", "sale_price", "floor_area_sqft", "satisfaction_score"]
    ].mean()
)

plt.figure(figsize=(8,6))
plt.scatter(
    data["sale_price"],
    data["floor_area_sqft"],
    c=data["cluster"]
)

plt.xlabel("Sale Price")
plt.ylabel("Floor Area (sqft)")
plt.title("Buyer Segments")
plt.show()

data.to_csv("buyer_segments.csv", index=False)

print("Buyer segments saved successfully!")

print(data["cluster"].value_counts())

cluster_summary = data.groupby("cluster")[
    ["age", "sale_price", "floor_area_sqft", "satisfaction_score"]
].mean()

print(cluster_summary)