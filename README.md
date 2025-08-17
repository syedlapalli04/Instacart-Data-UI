# Instacart Customer Segmentation Dashboard

This project is an interactive Streamlit dashboard for segmenting Instacart customers based on their order behavior. It provides visualizations, clustering, and insights to help understand customer groups and their characteristics.

## Features
- **Customer Grouping:** Segment customers using KMeans clustering (more algorithms coming soon).
- **Feature Selection:** Choose which behavioral features to use for clustering.
- **Visualizations:**
  - Pie chart of customer group sizes
  - Boxplots of feature distributions by group
  - Parallel coordinates plot with feature display names
  - Interactive group explorer and summary statistics
- **Professional UI:** Modern sidebar controls, clean layout, and informative headings.

## Data
The dashboard uses Instacart order data (CSV files):
- `data/orders.csv`
- `data/order_products__prior.csv`
- `data/order_products__train.csv`
- `data/products.csv`, `data/aisles.csv`, `data/departments.csv` (optional for further analysis)

## Usage
1. Clone this repository and place the Instacart CSV files in the `data/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
4. Use the sidebar to select clustering parameters and explore customer segments.

## Development Status
- Currently, only KMeans clustering is available for customer grouping.
- The dashboard is actively evolving; future updates will add more clustering algorithms and features.

## File Overview
- `app.py`: Main Streamlit dashboard
- `data_processing.py`: Data loading and feature engineering
- `clustering.py`: Clustering and dimensionality reduction logic

## License
This project is for educational and demonstration purposes.

## Author
Created by [Your Name].
