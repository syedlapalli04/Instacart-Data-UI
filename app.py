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

# Professional feature name mapping
feature_name_map = {
    "total_orders": "Total Orders",
    "avg_days_between_orders": "Avg. Days Between Orders",
    "avg_order_size": "Avg. Order Size",
    "avg_order_hour": "Avg. Order Hour",
    "avg_order_dow": "Most Common Day of Week",
    "reorder_ratio": "Reorder Ratio"
}

# For display only
viz_features = list(feature_name_map.keys())

st.title("Instacart Customer Data Analysis")
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

if st.session_state["page"] == "welcome":
    st.header("Welcome to Instacart Customer Data Analysis")
    st.markdown("""
    This dashboard helps you explore and segment Instacart customers based on their order behavior. 
    You can view customer groups, analyze their characteristics, and visualize patterns interactively.
    """)
    if st.button("Start Analysis"):
        st.session_state["page"] = "analysis"
    st.stop()

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

st.sidebar.header("Customer Grouping Parameters")
algorithm = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN"])
selected_features = st.sidebar.multiselect(
    "Features", [feature_name_map.get(f, f) for f in feature_cols], default=[feature_name_map.get(f, f) for f in feature_cols]
)

# Map selected features back to original names
selected_features_raw = [k for k, v in feature_name_map.items() if v in selected_features]
if not selected_features_raw:
    selected_features_raw = feature_cols

scaled, scaler = scale_features(features, selected_features_raw)

if algorithm == "KMeans":
    n_clusters = st.sidebar.slider("Number of Customer Groups", 2, 4, 4)
    labels, model = run_kmeans(scaled, n_clusters)
else:
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 20, 5)
    labels, model = run_dbscan(scaled, eps, min_samples)

features["customer_group"] = labels


# PCA for visualization
reduced, pca = run_pca(scaled, n_components=2)
features["pca1"] = reduced[:,0]
features["pca2"] = reduced[:,1]

# Increase sidebar width for better slider visibility
st.markdown("""<style>.css-1d391kg {min-width: 300px;}</style>""", unsafe_allow_html=True)

# Recommendations and profile names
customer_profiles = features.groupby("customer_group")[selected_features_raw].mean().reset_index()
if not customer_profiles.empty:
    def get_recommendation(row):
        # Use original column names for both row and features
        if row.get("total_orders", 0) > features["total_orders"].quantile(0.75):
            return "High-frequency buyers: Offer loyalty rewards, exclusive deals, and early access to new products."
        elif row.get("avg_order_size", 0) > features["avg_order_size"].quantile(0.75):
            return "Bulk shoppers: Provide bundle discounts, free shipping for large orders, and bulk promotions."
        elif row.get("avg_days_between_orders", 0) > features["avg_days_between_orders"].quantile(0.75):
            return "Occasional shoppers: Send re-engagement emails, personalized recommendations, and reminders."
        else:
            return "Average shoppers: Maintain regular promotions and highlight popular products."
    customer_profiles["recommendation"] = customer_profiles.apply(get_recommendation, axis=1)
    customer_profiles["profile_name"] = customer_profiles.apply(lambda row: f"Group {row['customer_group']} - {row['recommendation'].split(':')[0]}", axis=1)
    # Create mapping and add to features
    group_name_map = dict(zip(customer_profiles["customer_group"], customer_profiles["profile_name"]))
    features["customer_group_name"] = features["customer_group"].map(group_name_map)
    st.subheader("Customer Group Recommendations")
    st.write(customer_profiles[["customer_group", "recommendation"]])
else:
    st.subheader("Customer Group Recommendations")
    st.write("No customer group recommendations available.")

sns.set(font_scale=0.6) 


# st.subheader("Customer Group Scatter Plot (PCA)")
# fig, ax = plt.subplots()
# sns.scatterplot(x="pca1", y="pca2", hue="customer_group_name", palette="muted", data=features, ax=ax)
# st.pyplot(fig)


st.subheader("Customer Group Size Pie Chart")
group_counts = features["customer_group_name"].value_counts()
fig, ax = plt.subplots()
ax.pie(group_counts, labels=group_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
st.pyplot(fig)

st.subheader("Feature Distribution by Customer Group")
for feat in viz_features:
    fig, ax = plt.subplots()
    label = feature_name_map.get(feat, feat)
    sns.boxplot(x="customer_group_name", y=feat, data=features, palette="pastel", ax=ax)
    ax.set_xlabel("Customer Group")
    ax.set_ylabel(label)
    plt.tight_layout()
    st.pyplot(fig)


# st.subheader("Customer Group Size Pie Chart")
# group_counts = features["customer_group_name"].value_counts()
# fig, ax = plt.subplots()
# ax.pie(group_counts, labels=group_counts.index, autopct="%1.1f%%", colors=sns.color_palette("pastel"))
# st.pyplot(fig)


sns.set(font_scale=1.0) 

st.subheader("Parallel Coordinates Plot (Customer Group Averages)")
customer_means_profile = features.groupby("customer_group_name")[viz_features].mean().reset_index()
fig, ax = plt.subplots(figsize=(10,5))
parallel_coordinates(customer_means_profile, "customer_group_name", color=sns.color_palette("muted"))
st.pyplot(fig)

st.subheader("Interactive Customer Group Explorer")
selected_group = st.selectbox("Select Customer Group", sorted(features["customer_group_name"].unique()))
group_users = features[features["customer_group_name"] == selected_group]
st.write(group_users.head(10))

st.subheader("Customer Group Insights")
st.write("Customer Group Profiles:")
st.write(group_users[viz_features].describe())
