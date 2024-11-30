import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Function to load dataset


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to preprocess data


def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    # Impute missing values with the median of each column
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Scaling the data
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(
        df[numeric_cols]), columns=numeric_cols)

    return scaled_data, numeric_cols


# App layout
st.title("ClusterVision")

# File upload
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    # Load and display dataset
    df = load_data(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Preprocess data for clustering
    scaled_data, numeric_cols = preprocess_data(df)

    # Clustering settings
    st.write("### Clustering Settings")

    # Select columns for clustering
    st.write("Select columns for clustering (at least 2 columns):")
    selected_cols = st.multiselect(
        "Select columns", numeric_cols, default=numeric_cols[:2])

    if len(selected_cols) >= 2:
        # Use selected columns for clustering
        scaled_data = scaled_data[selected_cols]

        # Number of clusters slider
        k = st.slider("Select number of clusters (K)",
                      min_value=2, max_value=10, value=3)

        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Elbow method for optimal K
        st.write("### Elbow Method to Find Optimal K")
        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(k_range, inertia, marker='o')
        ax.set_xlabel("Number of Clusters (K)", fontsize=12)
        ax.set_ylabel("Inertia", fontsize=12)
        ax.set_title("Elbow Method", fontsize=14)
        st.pyplot(fig)

        # Silhouette score
        st.write("### Silhouette Score")
        silhouette_avg = silhouette_score(scaled_data, df['Cluster'])
        st.write(
            f"The average Silhouette Score for K={k} is: {silhouette_avg:.2f}")

        # Toggle between 2D and 3D visualization
        visualization_type = st.radio(
            "Select Visualization Type", ("2D", "3D"))

        # Display cluster plot based on the selection
        if visualization_type == "3D" and len(selected_cols) == 3:
            # 3D Cluster Visualization using Plotly
            st.write("### 3D Cluster Visualization")

            # Create the 3D scatter plot
            fig = px.scatter_3d(df, x=selected_cols[0], y=selected_cols[1], z=selected_cols[2],
                                color='Cluster', title="3D Cluster Visualization",
                                color_continuous_scale='Viridis', opacity=0.8)

            # Set axis labels and title
            fig.update_layout(scene=dict(
                xaxis_title=selected_cols[0],
                yaxis_title=selected_cols[1],
                zaxis_title=selected_cols[2]),
                margin=dict(l=0, r=0, b=0, t=0))

            # Show the plot
            st.plotly_chart(fig)
        elif visualization_type == "2D":
            # 2D Cluster Visualization
            st.write("### 2D Cluster Visualization")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(
                x=df[selected_cols[0]],
                y=df[selected_cols[1]],
                hue=df['Cluster'],
                palette="viridis",
                ax=ax,
                alpha=0.8
            )
            ax.set_title("Cluster Visualization", fontsize=14)
            ax.set_xlabel(selected_cols[0], fontsize=12)
            ax.set_ylabel(selected_cols[1], fontsize=12)
            st.pyplot(fig)

    else:
        st.write("Please select at least two columns for clustering.")

    # Hidden section for column distribution
    with st.expander("Show Column Distributions", expanded=False):
        if len(numeric_cols) > 0:
            selected_col = st.selectbox(
                "Select a column to visualize distribution:", numeric_cols)
            if selected_col:
                # Plot distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df[selected_col], kde=True, ax=ax, bins=30)
                ax.set_title(f"Distribution of {selected_col}", fontsize=14)
                ax.set_xlabel(selected_col, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                st.pyplot(fig)
        else:
            st.write("No numeric columns available for visualization.")
