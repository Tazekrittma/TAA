# Data Visualization and Clustering

## Dimensionality Reduction and Data Visualization
The "villes.csv" file contains data on 32 French cities, described by their average temperatures over the 12 months of the year. The objective of this section is to represent as much information as possible from this dataset graphically and potentially detect any topological segmentation of cities.

1. **Data Import**: We import the dataset using the Pandas library. The dataset includes temperature data for each month of the year for each city.
   
2. **Principal Component Analysis (PCA)**: We perform PCA on the dataset after centering and scaling it (StandardScaler). We determine the number of principal axes to retain to preserve at least 70% of the information in the initial data. We also provide an interpretation of the first two principal axes and visualize the cities projected onto the principal plane.

## Clustering

**Clustering** involves grouping objects (individuals or variables) into a limited number of classes or clusters, based on certain properties. In this section, we explore two clustering approaches, **K-Means** and **Agglomerative Clustering** (from the sklearn.cluster package), and apply them to the city dataset.

1. **K-Means Clustering**: We apply the K-Means procedure to the city dataset to obtain three clusters. We visualize the cities projected onto the principal plane, with each cluster represented by a different color.

2. **Agglomerative Clustering**: We apply the Agglomerative Clustering procedure to the city dataset to obtain three clusters using different aggregation methods (e.g., ward and average). Similar to K-Means, we visualize the results with different cluster colors.

## Determining the Optimal Number of Clusters
To determine the best partition (number of clusters) for the K-Means method, we employ the "Silhouette index" (metrics.silhouette_score from scikit-learn). We perform this analysis for various numbers of clusters (2, 3, 4, and 5) to identify the best partition with the highest Silhouette index.

For further details, please refer to the code.

**Note**: The code and explanations provided are part of a data analysis exercise for dimensionality reduction, data visualization, and clustering of city and temperature data. The primary goal is to explore the structure of the data and discover potential patterns.

**License**: MIT License
By Massinissa TAZEKRITT
