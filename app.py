"""
Streamlit UI for Instacart customer segmentation
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_data, aggregate_user_features
from clustering import scale_features, run_kmeans, run_dbscan, run_pca, get_silhouette
from pandas.plotting import parallel_coordinates

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

# Example cluster recommendations
# st.subheader("Cluster Recommendations")
def get_recommendation(row):
    # Example logic based on total_orders and avg_order_size
    if row.get("total_orders", 0) > features["total_orders"].quantile(0.75):
        return "High-frequency buyers: Offer loyalty rewards, exclusive deals, and early access to new products."
    elif row.get("avg_order_size", 0) > features["avg_order_size"].quantile(0.75):
        return "Bulk shoppers: Provide bundle discounts, free shipping for large orders, and bulk promotions."
    elif row.get("avg_days_between_orders", 0) > features["avg_days_between_orders"].quantile(0.75):
        return "Occasional shoppers: Send re-engagement emails, personalized recommendations, and reminders."
    else:
        return "Average shoppers: Maintain regular promotions and highlight popular products."

# Use cluster numbers for recommendations, then map to names
cluster_profiles = features.groupby("cluster")[selected_features].mean().reset_index()
if not cluster_profiles.empty:
    cluster_profiles["recommendation"] = cluster_profiles.apply(get_recommendation, axis=1)
    cluster_profiles["profile_name"] = cluster_profiles.apply(lambda row: f"{row['cluster']} - {row['recommendation'].split(':')[0]}", axis=1)
    # st.write(cluster_profiles[["cluster", "recommendation"]])
    # Create mapping from cluster number to profile name
    cluster_name_map = dict(zip(cluster_profiles["cluster"], cluster_profiles["profile_name"]))
    # Add profile name to features
    features["cluster_profile_name"] = features["cluster"].map(cluster_name_map)
else:
    st.write("No cluster profiles to recommend for.")
    cluster_name_map = {c: str(c) for c in features["cluster"].unique()}
    features["cluster_profile_name"] = features["cluster"].map(cluster_name_map)

# Now calculate cluster_means_profile for visualizations
cluster_means_profile = features.groupby("cluster_profile_name")[selected_features].mean().reset_index()

# Update all visualizations and dropdowns to use cluster_profile_name

sns.set(font_scale=0.5) 

st.subheader("Cluster Scatter Plot (PCA)")
fig, ax = plt.subplots()
sns.scatterplot(x="pca1", y="pca2", hue="cluster_profile_name", palette="muted", data=features, ax=ax)
st.pyplot(fig)

st.subheader("Feature Distribution by Cluster")
for feat in selected_features:
    fig, ax = plt.subplots()
    sns.boxplot(x="cluster_profile_name", y=feat, data=features, palette="pastel")
    #plt.tight_layout()
    # sns.set(font_scale=0.5) 
    st.pyplot(fig)

st.subheader("Cluster Size Pie Chart")
cluster_counts = features["cluster_profile_name"].value_counts()
fig, ax = plt.subplots()
ax.pie(cluster_counts, labels=cluster_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
st.pyplot(fig)

sns.set(font_scale=1.0) 

st.subheader("Parallel Coordinates Plot (Cluster Averages)")
fig, ax = plt.subplots(figsize=(10,5))
parallel_coordinates(cluster_means_profile, "cluster_profile_name", color=sns.color_palette("muted"))
st.pyplot(fig)

st.subheader("Interactive User Explorer")
selected_profile = st.selectbox("Select Cluster", sorted(features["cluster_profile_name"].unique()))
cluster_users = features[features["cluster_profile_name"] == selected_profile]
st.write(cluster_users.head(10))

st.subheader("Cluster Insights")
st.write("Cluster Profiles:")
st.write(cluster_users[selected_features].describe())

# Example cluster recommendations
st.subheader("Cluster Recommendations")
def get_recommendation(row):
    # Example logic based on total_orders and avg_order_size
    if row.get("total_orders", 0) > features["total_orders"].quantile(0.75):
        return "High-frequency buyers: Offer loyalty rewards, exclusive deals, and early access to new products."
    elif row.get("avg_order_size", 0) > features["avg_order_size"].quantile(0.75):
        return "Bulk shoppers: Provide bundle discounts, free shipping for large orders, and bulk promotions."
    elif row.get("avg_days_between_orders", 0) > features["avg_days_between_orders"].quantile(0.75):
        return "Occasional shoppers: Send re-engagement emails, personalized recommendations, and reminders."
    else:
        return "Average shoppers: Maintain regular promotions and highlight popular products."

# Use cluster numbers for recommendations, then map to names
cluster_profiles = features.groupby("cluster")[selected_features].mean().reset_index()
if not cluster_profiles.empty:
    cluster_profiles["recommendation"] = cluster_profiles.apply(get_recommendation, axis=1)
    # cluster_profiles["profile_name"] = cluster_profiles.apply(lambda row: f"{row['cluster']} - {row['recommendation'].split(':')[0]}", axis=1)
    st.write(cluster_profiles[["cluster", "recommendation"]])
    # Create mapping from cluster number to profile name
    # cluster_name_map = dict(zip(cluster_profiles["cluster"], cluster_profiles["profile_name"]))
    # Add profile name to features
    # features["cluster_profile_name"] = features["cluster"].map(cluster_name_map)
else:
    st.write("No cluster profiles to recommend for.")
    #cluster_name_map = {c: str(c) for c in features["cluster"].unique()}
    #features["cluster_profile_name"] = features["cluster"].map(cluster_name_map)
