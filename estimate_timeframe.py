import pandas as pd

orders = pd.read_csv("data/orders.csv")
orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0)
user_spans = orders.groupby('user_id')['days_since_prior_order'].sum()
total_days = user_spans.max()
print(f"Estimated maximum user order span: {total_days:.0f} days (~{total_days/365:.1f} years)")
