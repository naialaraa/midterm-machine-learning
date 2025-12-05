# Customer Clustering Analysis - Machine Learning Project

## 📋 Project Overview

This project implements a comprehensive end-to-end machine learning pipeline for customer segmentation using clustering algorithms. The analysis segments customers based on their spending and payment behavior patterns from credit card transaction data.

## 🎯 Objectives

- Design and implement a complete clustering pipeline
- Segment customers into meaningful groups
- Compare multiple clustering algorithms
- Provide actionable business insights for targeted marketing

## 📊 Dataset Information

- **Total Records**: 8,950 customers
- **Features**: 18 variables including:
  - Balance and payment information
  - Purchase behavior (one-off and installment)
  - Cash advance patterns
  - Credit limit and payment frequency
  - Customer tenure

## 🔧 Technologies Used

### Core Libraries
- **Polars** (v1.33.1+): High-performance data processing
- **Pandas**: Data manipulation and sklearn compatibility
- **NumPy**: Numerical computations
- **Scikit-learn**: Clustering algorithms and evaluation metrics
- **Matplotlib & Seaborn**: Data visualization
- **SciPy**: Statistical analysis and hierarchical clustering

### Clustering Algorithms
1. **K-Means**: Partitioning-based clustering
2. **Hierarchical Clustering**: Agglomerative with Ward linkage
3. **DBSCAN**: Density-based clustering with outlier detection

## 📁 Project Structure

```
UTS Dataset 3/
├── midterm_clustering_code.ipynb    # Main clustering pipeline notebook
├── Machine Learning/
│   └── clusteringmidterm.csv        # Customer dataset
├── customer_clusters_result.csv     # Output with cluster assignments
└── README.md                        # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install polars pandas numpy scikit-learn matplotlib seaborn scipy gdown
```

### Running the Analysis

1. **Download Dataset**:
   - Run cells 3-4 in the notebook to download data from Google Drive
   - Or place `clusteringmidterm.csv` in the `Machine Learning/` folder

2. **Execute the Pipeline**:
   - Open `midterm_clustering_code.ipynb` in Jupyter or VS Code
   - Run all cells sequentially
   - Estimated execution time: 3-5 minutes

3. **View Results**:
   - Clustered data will be saved to `customer_clusters_result.csv`
   - Visualizations appear inline in the notebook

## 📈 Pipeline Workflow

### 1. Data Loading & Exploration
- Load dataset using Polars for efficiency
- Initial data inspection and statistics

### 2. Data Cleaning & Preprocessing
- Handle missing values (median for numerical, mode for categorical)
- Remove duplicate records
- Data quality validation

### 3. Outlier Detection & Treatment
- IQR method for outlier identification
- Winsorization (capping) for outlier treatment
- Boxplot visualizations

### 4. Feature Engineering
- Domain-specific feature creation (if applicable)
- Feature selection for clustering

### 5. Exploratory Data Analysis (EDA)
- Distribution analysis of all features
- Correlation heatmap
- Feature relationship exploration

### 6. Feature Scaling
- StandardScaler normalization
- Prepare features for distance-based algorithms

### 7. Optimal Cluster Determination
- **Elbow Method**: WCSS analysis
- **Silhouette Score**: Cluster quality evaluation
- **Dendrogram**: Hierarchical structure visualization

### 8. Model Training
- Train three clustering algorithms
- Generate cluster assignments
- Visualize cluster distributions

### 9. Cluster Evaluation
- **Silhouette Score**: Cluster cohesion and separation
- **Davies-Bouldin Index**: Cluster similarity measure
- **Calinski-Harabasz Score**: Variance ratio criterion

### 10. Cluster Profiling
- Analyze cluster characteristics
- Generate cluster statistics
- Create heatmaps for feature comparison

### 11. Visualization
- **PCA 2D/3D plots**: Dimensionality reduction for visualization
- Compare clustering results across algorithms
- Interactive cluster exploration

### 12. Results Export
- Save clustered data with segment labels
- Include all algorithm results
- Ready for business analysis

## 📊 Evaluation Metrics

### Silhouette Score
- **Range**: -1 to 1
- **Interpretation**: Higher is better (>0.5 indicates good clustering)
- **Measures**: How similar points are to their cluster vs. other clusters

### Davies-Bouldin Index
- **Range**: 0 to ∞
- **Interpretation**: Lower is better (0 is ideal)
- **Measures**: Average similarity between clusters

### Calinski-Harabasz Score
- **Range**: 0 to ∞
- **Interpretation**: Higher is better
- **Measures**: Ratio of between-cluster to within-cluster variance

## 🎯 Key Findings

The analysis identifies distinct customer segments with different:
- Spending patterns
- Payment behaviors
- Credit utilization
- Purchase frequencies
- Cash advance usage

*(Specific findings will appear after running the complete analysis)*

## 💡 Business Applications

1. **Targeted Marketing**: Customize campaigns for each segment
2. **Product Recommendations**: Offer relevant products based on cluster characteristics
3. **Risk Management**: Identify high-risk customer segments
4. **Customer Retention**: Develop strategies to retain valuable customers
5. **Credit Limit Optimization**: Adjust limits based on segment behavior

## 📝 Output Files

### customer_clusters_result.csv
Contains original features plus cluster assignments from all algorithms:
- `KMeans_Cluster`: K-Means cluster assignment
- `Hierarchical_Cluster`: Hierarchical clustering assignment
- `DBSCAN_Cluster`: DBSCAN assignment (-1 indicates noise/outliers)
- `Customer_Segment`: Business-friendly segment labels

## ⚡ Performance Optimization

- **Polars** used for efficient data operations (10-100x faster than Pandas)
- Conversion to Pandas only when necessary for sklearn compatibility
- Memory-efficient processing for large datasets
- Optimized visualization rendering

## 🔍 Algorithm Selection Guide

| Algorithm | Best For | Advantages | Limitations |
|-----------|----------|------------|-------------|
| **K-Means** | Spherical clusters | Fast, scalable | Requires K specification |
| **Hierarchical** | Hierarchical structure | No K needed upfront | Computationally expensive |
| **DBSCAN** | Irregular shapes, outliers | Finds arbitrary shapes | Sensitive to parameters |

## 📚 References

- Dataset: Credit Card Customer Data
- Scikit-learn Documentation: https://scikit-learn.org/
- Polars Documentation: https://pola.rs/
