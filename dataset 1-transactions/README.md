# 🛡️ Fraud Detection in Online Transactions

## 📋 Project Overview

**Course**: Machine Learning - Midterm Assignment  
**Objective**: Design and implement an end-to-end machine learning pipeline to predict the probability of online transactions being fraudulent.

### 🎯 Key Challenge
- **Severe class imbalance**: Only ~3.5% of transactions are fraudulent
- **High dimensionality**: 393 features per transaction
- **Large dataset**: 590K+ training samples, 506K+ test samples
- **Memory constraints**: Requires efficient data processing

---

## 📊 Dataset

### Data Source
- **Google Drive ID**: `1JvI5xhPfN3VmjpWYZk9fCHG41xG697um`
- **Files**:
  - `train_transaction.csv`: 590,540 transactions × 394 columns (393 features + `isFraud` target)
  - `test_transaction.csv`: 506,691 transactions × 393 features

### Features
- **Transaction Details**: Transaction amounts, time stamps, transaction types
- **Card Information**: Card types, card categories, anonymized card features
- **Address Features**: Billing/shipping address information
- **Device Information**: Device type, OS, browser details
- **Email Domain**: Email provider information
- **Product Information**: Product codes and categories
- **Anonymized Features**: Multiple C1-C14, D1-D15, V1-V339 features

### Target Variable
- `isFraud`: Binary classification (0 = legitimate, 1 = fraud)
- **Imbalance Ratio**: ~96.5% legitimate vs ~3.5% fraudulent

---

## 🔧 Technical Stack

### Libraries & Frameworks
```python
# Data Processing
- polars          # Memory-efficient data loading (20-30% less RAM than pandas)
- pandas          # sklearn compatibility
- numpy           # Numerical operations

# Machine Learning
- scikit-learn    # ML models, preprocessing, evaluation
- xgboost         # Gradient boosting (CPU only)
- imbalanced-learn # SMOTE for handling class imbalance

# Visualization
- matplotlib      # Plotting
- seaborn         # Statistical visualizations
```

### Environment
- **Python**: 3.11+
- **OS**: Windows
- **Memory Optimization**: Polars for data loading, Pandas for sklearn operations

---

## 🚀 Pipeline Workflow

### 1️⃣ **Data Loading & Exploration**
- Download dataset from Google Drive using `gdown`
- Load data with **Polars** for memory efficiency
- Display memory usage comparison
- Initial data inspection (shape, types, statistics)

### 2️⃣ **Exploratory Data Analysis (EDA)**
Comprehensive visualizations including:
- ✅ Target distribution analysis (fraud vs legitimate)
- ✅ Transaction amount patterns
- ✅ Correlation heatmap for top features
- ✅ Categorical feature analysis (ProductCD, card types, email domains)
- ✅ Time-based fraud patterns (hourly transaction volume & fraud rates)
- ✅ Transaction amount distribution by fraud status

### 3️⃣ **Data Preprocessing**

#### Feature Separation
```python
X = train_df.drop('isFraud', axis=1)  # Features
y = train_df['isFraud']                # Target
X_test_final = test_df.copy()          # Test features
```

#### Missing Value Handling
- **Numerical features**: Fill with median
- **Categorical features**: Fill with mode (most frequent value)

#### Feature Selection
- **Variance Threshold**: Remove low-variance features (threshold=0.01)
- Reduces dimensionality while preserving important patterns

#### Encoding
- **Label Encoding**: Convert categorical variables to numerical format
- Applied to all object-type columns

#### Feature Scaling
- **StandardScaler**: Normalize features (mean=0, std=1)
- Essential for distance-based algorithms

#### Train-Test Split
- **Split ratio**: 80% training, 20% validation
- **Stratify**: Maintain class distribution in both sets
- **Random state**: 42 (reproducibility)

### 4️⃣ **Handling Class Imbalance**
- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Strategy**: Generate synthetic samples for minority class (fraud)
- **Random state**: 42
- **Result**: Balanced training set for better model learning

### 5️⃣ **Model Training**

Four machine learning algorithms evaluated:

#### 1. Logistic Regression
- Simple, interpretable baseline model
- Fast training and prediction
- Good for linearly separable data

#### 2. Decision Tree
- Non-linear decision boundaries
- Easy to visualize and interpret
- Prone to overfitting (controlled with max_depth=10)

#### 3. Random Forest
- Ensemble of decision trees
- Robust to overfitting
- **Hyperparameters**: 100 estimators, max_depth=20
- Feature importance analysis

#### 4. XGBoost (CPU)
- Gradient boosting ensemble
- State-of-the-art performance
- **Hyperparameters**: 
  - Learning rate: 0.1
  - Max depth: 7
  - Estimators: 100
  - Tree method: 'hist' (CPU optimized)
  - Eval metric: 'logloss'

### 6️⃣ **Hyperparameter Tuning**

**GridSearchCV** on Random Forest (best performing model):
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```
- **Cross-validation**: 3-fold CV
- **Metric**: ROC-AUC score
- **Jobs**: -1 (parallel processing)

### 7️⃣ **Model Evaluation**

#### Metrics Computed
- ✅ **Accuracy**: Overall correctness
- ✅ **Precision**: True fraud / Predicted fraud (minimize false alarms)
- ✅ **Recall**: True fraud / Actual fraud (catch all fraud cases)
- ✅ **F1-Score**: Harmonic mean of precision and recall
- ✅ **ROC-AUC**: Area under ROC curve (threshold-independent metric)

#### Visualization
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **ROC Curve**: TPR vs FPR for all models
- **Performance Comparison**: Bar charts for all metrics

### 8️⃣ **Model Selection & Final Predictions**
- Select best performing model based on ROC-AUC and F1-score
- Generate predictions on test dataset
- Create submission file: `submission.csv` with columns [`TransactionID`, `isFraud`]

---

## 📈 Expected Results

### Performance Metrics (Typical Range)
- **Logistic Regression**: ROC-AUC ~0.85-0.90
- **Decision Tree**: ROC-AUC ~0.75-0.85
- **Random Forest**: ROC-AUC ~0.90-0.95
- **XGBoost**: ROC-AUC ~0.92-0.96 (Best)

### Key Insights
- XGBoost typically performs best due to gradient boosting
- SMOTE significantly improves recall on minority class
- Feature selection reduces overfitting and speeds up training
- Polars reduces memory usage by 20-30% compared to Pandas

---

## 💻 Usage

### 1. Setup Environment
```bash
# Install required packages
pip install polars pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn gdown
```

### 2. Run the Notebook
```bash
# Open in Jupyter or VS Code
jupyter notebook midterm_transaction_data.ipynb
```

### 3. Execute Cells Sequentially
1. **Install & Import**: Run cells 1-3
2. **Download Data**: Run cells 4-5 (downloads from Google Drive)
3. **EDA**: Run cells 6-23 (exploratory analysis & visualizations)
4. **Preprocessing**: Run cells 24-38 (feature engineering, encoding, scaling)
5. **Model Training**: Run cells 39-51 (train 4 models)
6. **Hyperparameter Tuning**: Run cells 52-54 (GridSearch on best model)
7. **Evaluation**: Run cells 55-59 (metrics, visualizations, comparison)
8. **Predictions**: Run cells 60-61 (generate submission file)

### 4. Output Files
- `submission.csv`: Final predictions for test data

---

## 🎓 Key Learnings

### Technical Skills
- ✅ End-to-end ML pipeline implementation
- ✅ Handling imbalanced datasets with SMOTE
- ✅ Feature engineering and selection techniques
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Model comparison and evaluation
- ✅ Memory-efficient data processing with Polars

### Machine Learning Concepts
- ✅ Classification algorithms (Linear, Tree-based, Ensemble)
- ✅ Cross-validation strategies
- ✅ Evaluation metrics for imbalanced data
- ✅ Overfitting prevention (feature selection, ensemble methods)
- ✅ Model interpretability vs performance trade-offs

### Best Practices
- ✅ Reproducibility (random_state=42)
- ✅ Memory optimization (Polars vs Pandas)
- ✅ Code documentation and visualization
- ✅ Systematic model comparison
- ✅ Train-test separation to prevent data leakage

---

## 📝 Notes

### Memory Optimization
- **Polars** is used for data loading: 20-30% less RAM compared to Pandas
- Data is converted to Pandas only when needed for sklearn operations
- Garbage collection (`gc.collect()`) used to free memory

### Class Imbalance Strategy
- SMOTE applied only on training set (not validation/test)
- Evaluation metrics focused on recall and ROC-AUC (not just accuracy)
- Confusion matrix analysis to understand false positives vs false negatives

### Feature Engineering
- Time-based features extracted for analysis only (not included in model)
- Categorical encoding applied consistently to train and test
- Missing value imputation strategy documented

### Model Selection Criteria
1. **Primary**: ROC-AUC score (threshold-independent)
2. **Secondary**: F1-score (balance precision-recall)
3. **Consideration**: Training time and model complexity

---

## 👨‍💻 Author

**Semester 7 - Machine Learning Course**  
**Institution**: [Your University Name]  
**Date**: December 2025

---

## 📄 License

This project is for educational purposes as part of a university midterm assignment.

---

## 🙏 Acknowledgments

- Dataset provided by course instructor
- scikit-learn and XGBoost documentation
- Polars documentation for memory-efficient data processing
- imbalanced-learn library for SMOTE implementation

---

## 📧 Contact

For questions regarding this project, please contact through university channels.
