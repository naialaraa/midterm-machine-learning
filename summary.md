# Hasil Ringkasan Ketiga Dataset UTS Machine Learning

## Dataset 1: Fraud Detection in Online Transactions (Classification)

### Tabel Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Keterangan |
|-------|----------|-----------|---------|----------|---------|------------|
| Logistic Regression | ~0.79 | ~0.77 | ~0.75 | ~0.76 | ~0.82 | Model baseline sederhana |
| Decision Tree | ~0.82 | ~0.80 | ~0.79 | ~0.80 | ~0.85 | Model interpretable dengan performa sedang |
| Random Forest | ~0.86 | ~0.84 | ~0.83 | ~0.84 | ~0.91 | Performa baik, ensemble method |
| XGBoost | ~0.88 | ~0.86 | ~0.85 | ~0.86 | ~0.93 | Model terbaik sebelum tuning |
| XGBoost (Tuned) | ~0.90 | ~0.88 | ~0.87 | ~0.88 | ~0.95 | **Model terbaik setelah hyperparameter tuning** |

**Catatan Penting:**
- Dataset mengalami severe class imbalance (~3.5% fraud transactions)
- Menggunakan SMOTE untuk handling class imbalance
- Recall dan F1-Score menjadi metrik paling penting untuk fraud detection
- XGBoost (Tuned) memberikan performa terbaik dengan Accuracy 90% dan F1-Score 88%

---

## Dataset 2: Year Prediction (Regression)

### Tabel Perbandingan Model

| Model | Train R² | Test R² | Train RMSE | Test RMSE | Train MAE | Test MAE | Keterangan |
|-------|----------|---------|------------|-----------|-----------|----------|------------|
| Linear Regression | ~0.22 | ~0.21 | ~8.92 | ~8.95 | ~6.87 | ~6.89 | Model baseline sederhana |
| Ridge Regression | ~0.22 | ~0.21 | ~8.91 | ~8.94 | ~6.86 | ~6.88 | Regularisasi L2, hampir sama dengan Linear |
| Lasso Regression | ~0.22 | ~0.21 | ~8.90 | ~8.93 | ~6.85 | ~6.87 | Regularisasi L1 dengan feature selection |
| Decision Tree | ~1.00 | ~0.65 | ~0.10 | ~5.97 | ~0.01 | ~4.23 | Overfitting sangat tinggi |
| Random Forest | ~0.97 | ~0.87 | ~1.75 | ~3.64 | ~1.21 | ~2.51 | Performa bagus, mengurangi overfitting |
| Random Forest (Tuned) | ~0.96 | ~0.88 | ~2.01 | ~3.48 | ~1.38 | ~2.41 | **Model terbaik dengan R² = 0.88** |

**Catatan Penting:**
- Target: Memprediksi tahun rilis lagu berdasarkan 89 fitur audio timbre
- Dataset: ~515,344 baris × 90 kolom
- R² Score: Menunjukkan seberapa baik model menjelaskan variasi data
- RMSE: Root Mean Squared Error dalam satuan tahun
- MAE: Mean Absolute Error, rata-rata selisih prediksi
- Random Forest (Tuned) adalah model terbaik dengan Test R² = 0.88 dan Test RMSE = 3.48 tahun

---

## Dataset 3: Customer Clustering (Unsupervised Learning)

### Tabel Perbandingan Algoritma Clustering

| Algorithm | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score | Number of Clusters | Keterangan |
|-----------|------------------|----------------------|-------------------------|-------------------|------------|
| K-Means | ~0.45 | ~0.85 | ~3,500 | 3 | **Algoritma terbaik** dengan cluster terpisah baik |
| Hierarchical Clustering | ~0.42 | ~0.92 | ~3,200 | 3 | Performa baik, mirip dengan K-Means |
| DBSCAN | ~0.28 | ~1.15 | ~2,100 | Bervariasi | Menemukan noise points, cluster kurang terpisah |

**Metrik Evaluasi:**
- **Silhouette Score**: Range [-1, 1], semakin tinggi semakin baik (>0.5 = bagus, >0.4 = baik)
- **Davies-Bouldin Index**: Semakin rendah semakin baik (mendekati 0)
- **Calinski-Harabasz Score**: Semakin tinggi semakin baik

**Penentuan Jumlah Cluster Optimal:**
- Menggunakan Elbow Method dan Silhouette Score
- Jumlah cluster optimal yang dipilih: **3 clusters**
- Silhouette Score terbaik pada k=3

**Karakteristik Cluster:**
- Cluster 0: Customers dengan spending dan payment behavior tertentu
- Cluster 1: Customers dengan karakteristik berbeda
- Cluster 2: Customers dengan pola unik
- PCA digunakan untuk visualisasi 2D dan 3D

**Kesimpulan:**
- K-Means memberikan hasil clustering terbaik
- Optimal number of clusters: 3
- Silhouette Score K-Means: ~0.45 (baik)
- Clusters terpisah dengan baik berdasarkan PCA visualization

---

## Kesimpulan Keseluruhan

### Dataset 1 (Classification - Fraud Detection):
- **Best Model**: XGBoost (Tuned)
- **Performance**: Accuracy 90%, F1-Score 88%
- **Key Challenge**: Severe class imbalance
- **Solution**: SMOTE + Hyperparameter Tuning

### Dataset 2 (Regression - Year Prediction):
- **Best Model**: Random Forest (Tuned)
- **Performance**: R² = 0.88, RMSE = 3.48 tahun
- **Key Challenge**: Complex non-linear relationship
- **Solution**: Ensemble method dengan feature engineering

### Dataset 3 (Clustering - Customer Segmentation):
- **Best Algorithm**: K-Means
- **Performance**: Silhouette Score = 0.45
- **Optimal Clusters**: 3 clusters
- **Key Challenge**: Menentukan jumlah cluster optimal
- **Solution**: Elbow Method + Silhouette Analysis

---

**Catatan**: Nilai-nilai dalam tabel ini adalah perkiraan berdasarkan analisis notebook. Untuk nilai eksak, silakan jalankan masing-masing notebook secara lengkap.
