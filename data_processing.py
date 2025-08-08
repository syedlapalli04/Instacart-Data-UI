"""
Data processing for Instacart customer segmentation
"""
import pandas as pd
import numpy as np

def load_data(orders_path, prior_path, train_path):
    orders = pd.read_csv(orders_path)
    prior = pd.read_csv(prior_path)
    train = pd.read_csv(train_path)
    return orders, prior, train

def aggregate_user_features(orders, prior, train):
    # Combine prior and train order products
    order_products = pd.concat([prior, train], ignore_index=True)
    # Merge order_products with orders to get user_id for each product
    order_products = order_products.merge(orders[['order_id', 'user_id']], on='order_id', how='left')
    # Total orders per user
    user_orders = orders.groupby('user_id').order_number.max().rename('total_orders')
    # Avg days between orders
    avg_days = orders.groupby('user_id').days_since_prior_order.mean().fillna(0).rename('avg_days_between_orders')
    # Avg order size
    order_sizes = order_products.groupby('order_id').size()
    avg_order_size = orders.set_index('order_id').join(order_sizes.rename('order_size')).groupby('user_id').order_size.mean().rename('avg_order_size')
    # Avg order hour
    avg_hour = orders.groupby('user_id').order_hour_of_day.mean().rename('avg_order_hour')
    # Most common day of week
    mode_dow = orders.groupby('user_id').order_dow.agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).rename('avg_order_dow')
    # Reorder ratio
    if 'reordered' in order_products.columns:
        reorder_ratio = order_products.groupby('user_id').reordered.mean().rename('reorder_ratio')
    else:
        reorder_ratio = pd.Series(dtype=float)
    # Merge all features
    features = pd.concat([user_orders, avg_days, avg_order_size, avg_hour, mode_dow, reorder_ratio], axis=1)
    return features.reset_index()
