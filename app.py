"""
Streamlit UI for Instacart customer segmentation
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, aggregate_user_features
from clustering import scale_features, run_kmeans, run_dbscan, run_pca, get_silhouette

st.set_page_config(page_title="Instacart Customer Segmentation", layout="wide")
st.markdown("""
<style>
:root {
  --main-color: #b39ddb;
}
[data-testid="stSidebar"] {
  background-color: #f3eaff;
}
.stApp {
  background-color: #f8f6fc;
}
</style>
""", unsafe_allow_html=True)

st.title("Instacart Customer Segmentation")
st.write("Segment customers based on their order behavior and explore clusters interactively.")

# File paths (update as needed)
orders_path = "data/orders.csv"
prior_path = "data/order_products__prior.csv"
train_path = "data/order_products__train.csv"

@st.cache_data
def get_features():
    orders, prior, train = load_data(orders_path, prior_path, train_path)
    features = aggregate_user_features(orders, prior, train)
    return features

features = get_features()
feature_cols = [col for col in features.columns if col not in ["user_id"]]

st.sidebar.header("Clustering Parameters")
algorithm = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN"])
selected_features = st.sidebar.multiselect("Features", feature_cols, default=feature_cols)

scaled, scaler = scale_features(features, selected_features)

if algorithm == "KMeans":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    labels, model = run_kmeans(scaled, n_clusters)
else:
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)
    labels, model = run_dbscan(scaled, eps, min_samples)

features["cluster"] = labels

# PCA for visualization
reduced, pca = run_pca(scaled, n_components=2)
features["pca1"] = reduced[:,0]
features["pca2"] = reduced[:,1]

st.subheader("Cluster Scatter Plot (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(x="pca1", y="pca2", hue="cluster", palette="muted", data=features, ax=ax)
st.pyplot(fig)

st.subheader("Feature Distribution by Cluster")
for feat in selected_features:
    fig, ax = plt.subplots()
    sns.boxplot(x="cluster", y=feat, data=features, palette="pastel")
    st.pyplot(fig)

st.subheader("Cluster Size Pie Chart")
cluster_counts = features["cluster"].value_counts()
fig, ax = plt.subplots()
ax.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
st.pyplot(fig)

st.subheader("Parallel Coordinates Plot")
from pandas.plotting import parallel_coordinates
fig, ax = plt.subplots(figsize=(10,5))
parallel_coordinates(features[["cluster"] + selected_features], "cluster", color=sns.color_palette("muted"))
st.pyplot(fig)

st.subheader("Interactive User Explorer")
selected_cluster = st.selectbox("Select Cluster", sorted(features["cluster"].unique()))
cluster_users = features[features["cluster"] == selected_cluster]
st.write(cluster_users.head(10))

st.subheader("Cluster Insights")
st.write("Cluster Profiles:")
st.write(cluster_users[selected_features].describe())

sil_score = get_silhouette(scaled, labels)
if sil_score:
    st.write(f"Silhouette Score: {sil_score:.2f}")
