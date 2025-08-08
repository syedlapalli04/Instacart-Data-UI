"""
Clustering logic for Instacart customer segmentation
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler

def run_kmeans(scaled_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    return labels, kmeans

def run_dbscan(scaled_data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_data)
    return labels, dbscan

def run_pca(scaled_data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled_data)
    return reduced, pca

def get_silhouette(scaled_data, labels):
    if len(set(labels)) > 1:
        return silhouette_score(scaled_data, labels)
    return None
