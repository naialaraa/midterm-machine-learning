# Midterm Machine Learning

## 1. Tujuan Repositori

Repositori ini dibuat bertujuan untuk memenuhi tugas UTS mata kuliah Machine Learning. Tujuannya adalah membangun end-to-end pipeline yang mencakup pembacaan data, preprocessing, pemodelan, evaluasi, dan interpretasi hasil sehingga menghasilkan analisis yang dapat dipertanggungjawabkan.

## 2. Gambaran Umum Proyek

Proyek ini berisi eksperimen pada beberapa dataset yang diberikan oleh dosen. Setiap eksperimen mengikuti alur kerja standar machine learning:

- Eksplorasi data (Exploratory Data Analysis / EDA)
- Pembersihan dan transformasi data
- Pembagian data menjadi train dan test
- Pembuatan model baseline dan model lanjutan
- Evaluasi performa menggunakan metrik yang sesuai
- Visualisasi hasil dan penarikan kesimpulan

Semua pengerjaan dilakukan menggunakan Python dan Jupyter Notebook.

## 3. Struktur Repositori
```
midterm-machine-learning/
│
├── dataset_1/
│   ├── notebook_dataset1.ipynb
│
├── dataset_2/
│   ├── notebook_dataset2.ipynb
│
├── dataset_3/
│   ├── notebook_dataset3.ipynb
│
├── summary.md
│
└── README.md
```

## 4. Cara Menjalankan

### A. Clone repositori
```bash
git clone https://github.com/naialaraa/midterm-machine-learning.git
cd midterm-machine-learning
```

### B. Siapkan environment (opsional tetapi disarankan)
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### C. Install dependensi
```bash
pip install -r requirements.txt

# atau jika tidak ada requirements.txt:
pip install numpy pandas scikit-learn matplotlib seaborn jupyter notebook
```

### D. Jalankan Jupyter Notebook
```bash
jupyter notebook
```

Buka notebook sesuai folder dataset dan jalankan sel secara berurutan.

## 5. Isi dan Alur Kerja Notebook

Setiap notebook umumnya mengandung tahapan berikut:

### 1. Import library
Memuat paket utama: numpy, pandas, scikit-learn, matplotlib, seaborn, dll.

### 2. Load data
Membaca file data (CSV/Excel) dan menampilkan ringkasan awal.

### 3. Exploratory Data Analysis (EDA)
- Statistik deskriptif
- Distribusi fitur
- Korelasi antar fitur
- Pengecekan nilai hilang (missing values) dan anomali (outliers)

### 4. Preprocessing
- Penanganan missing values
- Encoding variabel kategori (LabelEncoder / OneHotEncoder)
- Skalasi fitur bila perlu (StandardScaler / MinMaxScaler)
- Pembagian data train-test
- Penanganan class imbalance jika diperlukan (oversampling/undersampling)

### 5. Feature engineering (opsional)
- Pembuatan fitur baru
- Seleksi fitur (feature selection)

### 6. Pelatihan model
- Model baseline (mis. Logistic Regression, Linear Regression)
- Model lanjutan (Decision Tree, Random Forest, XGBoost jika tersedia)
- Melakukan cross-validation dan tuning hyperparameter bila relevan

### 7. Evaluasi
- Mengukur performa dengan metrik yang sesuai (lihat tabel di bawah)
- Menampilkan confusion matrix, kurva pembelajaran, atau plot prediksi vs aktual

### 8. Visualisasi & Interpretasi
- Plot fitur penting (feature importance)
- Reduksi dimensi untuk visualisasi (PCA / t-SNE) untuk clustering

### 9. Kesimpulan dan rekomendasi
- Ringkasan temuan
- Rekomendasi perbaikan atau eksperimen lanjutan

## 6. Metrik Evaluasi yang Digunakan

| Tipe Problem | Metrik Utama |
|--------------|--------------|
| Klasifikasi | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Regresi | MAE, MSE, RMSE, R² Score |
| Clustering | Inertia / SSE, Silhouette Score, Visualisasi klaster (PCA/t-SNE) |

## 7. Contoh Model yang Digunakan

- Logistic Regression
- Decision Tree
- Random Forest
- Linear Regression (untuk regresi)
- K-Means / DBSCAN (untuk clustering)
- XGBoost

## 8. Hasil Ringkasan

├── summary.md

## 9. Identitas 

| Field | Value |
|-------|-------|
| Nama | Naia Lara Shafir Anwar |
| Kelas | TK-46-GAB |
| NIM | 1103223030 |
| Mata Kuliah | Machine Learning |
