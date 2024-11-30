# Customer Segmentation Project

This project focuses on customer segmentation using clustering techniques. The dataset contains information about customers' demographics, annual income, and spending scores. The goal is to identify distinct customer groups and analyze their behaviors.

## Features

- **Data Cleaning and Preprocessing**: Handles missing values and standardizes numerical columns.
- **Exploratory Data Analysis (EDA)**: Visualizes distributions and relationships in the data.
- **KMeans Clustering**: Groups customers into clusters based on their income and spending patterns.
- **Interactive Web App**: A Streamlit app to perform clustering on custom datasets.

---

## File Descriptions

### `main.ipynb`

- A Jupyter notebook for data analysis and customer segmentation.
- Performs:
  - Data cleaning and preprocessing.
  - Visualization of income and spending score distributions.
  - Clustering analysis using the KMeans algorithm.
  - Elbow method and silhouette scores to determine optimal clusters.

### `app.py`

- A Streamlit-based interactive web application.
- Allows users to:
  - Upload a custom dataset.
  - Select columns for clustering.
  - Visualize clusters in 2D or 3D.
  - View data distributions interactively.

### `requirements.txt`

- Contains the dependencies required to run the project. Install them using:
  ```bash
  pip install -r requirements.txt
  ```
