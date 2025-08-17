
# Instacart Customer Insights Dashboard

This project is an interactive web app that helps you explore and understand Instacart shoppers based on their buying habits. It shows easy-to-read charts and groupings to help you see patterns in how people shop.

## What Can You Do?
- **See Shopper Groups:** The app organizes shoppers into groups with similar shopping styles (using a smart computer method called KMeans for now, with more coming soon).
- **Pick What to Compare:** Choose which shopping habits (like order size or time of day) you want to look at.
- **View Visual Summaries:**
  - Pie chart showing the size of each shopper group
  - Boxplots showing how different groups shop
  - Colorful line chart comparing group averages
  - Browse and compare details for each group
- **Simple Controls:** Use the sidebar to easily change what you see and compare.

## Data Needed
Note : The files should be stored in the data folder, but the CSV files are too large to directly upload to GitHub. If you would like access to the data, you can:
- Download the necessary files from Kaggle from the following link: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset
- Contact me and I will send a zip file containing the three files needed

The app uses Instacart order data in CSV format:
- `data/orders.csv`
- `data/order_products__prior.csv`
- `data/order_products__train.csv`
- `data/products.csv`, `data/aisles.csv`, `data/departments.csv` (optional for extra details)


## How to Use
1. Download or clone this folder and put the Instacart CSV files in the `data/` folder.
2. Install the needed packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Use the sidebar to pick how you want to group and compare shoppers, and explore the results.

## Sidebar Options
The sidebar lets you customize what you see and how shoppers are grouped:

- **Algorithm:** Choose the method for grouping shoppers (currently only KMeans is available).
- **Number of Customer Groups:** Pick how many groups you want to create.
- **Features for Grouping:** Select which shopping habits (like order size, time of day, etc.) to use for making groups.
- **Show Plots/Tables With:** Decide if you want to see all features or just the ones you selected.
- **Boxplot Display Options:** Choose how boxplots are shown—include or hide outliers, or hide boxplots entirely.

These options help you explore the data in different ways and find patterns in how people shop.

## What's Inside
- `app.py`: The main dashboard
- `data_processing.py`: Loads and prepares the data
- `clustering.py`: Creates shopper groups and helps with charts

## License
This project is for learning and demonstration only.
