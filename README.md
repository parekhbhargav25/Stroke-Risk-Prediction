# 🧠 IntroToML(ECE 457B|657) Project — Stroke Risk Prediction 🩺

## 📊 Competition Overview

This project addresses the critical healthcare challenge of predicting stroke risk using machine learning. Strokes are a leading cause of death and disability worldwide, making early prediction and intervention crucial for patient outcomes.

**Competition Results:**
- **AUC Score: 0.87141** 🎯
- **Ranking: 10th place** out of 175 submissions 🏆
- **Competition:** IntroToML(ECE 457B|657) Kaggle Competition
- **Course:** Introduction to Machine Learning (ECE 457B/657)
- **Institution:** University of Waterloo
- **Instructor:** Professor Amirhossein Karimi
- **TA & Competition Lead:** Arezoo Alipanah

## 🎯 Project Objectives

This capstone challenge implements a complete, end-to-end machine learning pipeline to tackle a critical healthcare problem with real-world impact. The goal is to develop a robust machine learning model that can accurately predict the likelihood of a stroke based on various patient characteristics and medical indicators. This predictive model can assist healthcare professionals in identifying high-risk patients for early intervention and preventive care.

### Key Challenges Addressed:
- **Severe Class Imbalance:** Fewer than 5% of patients had a stroke, requiring robust handling techniques
- **Missing Data:** Strategic imputation of missing BMI values
- **Mixed Data Types:** Combination of numerical and categorical features requiring sophisticated preprocessing
- **Real-World Impact:** Direct application in preventative medicine

## 📈 Key Features & Methodology

### Data Analysis & Preprocessing
- **Dataset:** ~5,000 samples with 11 features (Stroke Prediction Dataset)
- **Target Variable:** Binary classification (stroke: 0/1)
- **Class Imbalance:** Severe imbalance (fewer than 5% positive cases)
- **Features:** Age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status
- **Missing Data:** Strategic handling of missing BMI values

### Machine Learning Pipeline

#### 1. **Baseline Models**
- Decision Tree Classifier (AUC: ~0.55)
- Logistic Regression with balanced class weights
- K-Nearest Neighbors (k=25, distance-weighted)

#### 2. **Advanced Model: CatBoost**
- **Final Model:** CatBoost Classifier
- **Key Optimizations:**
  - Class weight balancing for imbalanced data
  - Categorical feature handling
  - Early stopping to prevent overfitting
  - Hyperparameter tuning (learning_rate=0.03, depth=6, iterations=1500)

#### 3. **Feature Engineering**
- BMI missing value indicator
- Robust preprocessing pipeline with:
  - Median imputation for numerical features
  - One-hot encoding for categorical features
  - Standard scaling for distance-based algorithms
- **ColumnTransformer Pipeline:** Consistent handling of mixed data types

### Model Performance

| Model | AUC Score | Notes |
|-------|-----------|-------|
| Decision Tree (Baseline) | 0.554 | Initial baseline |
| Logistic Regression | ~0.73 | With class balancing |
| CatBoost (Final) | **0.8565** | 5-fold CV |
| **Final Submission** | **0.87141** | **10th place** |

## 🛠️ Technical Implementation

### Course Skills Demonstrated
This project showcases proficiency in:
- **Data cleaning, EDA, and missing-value imputation**
- **Advanced feature engineering and preprocessing with ColumnTransformer**
- **Model selection across non-parametric, linear, and non-linear algorithms**
- **Ensembling techniques and Gradient Boosting**
- **Cross-validation and error analysis**

### Environment Setup
```bash
pip install scikit-learn==1.4.2 xgboost==2.0.3 pandas==2.2.2 seaborn==0.13.2 catboost
```

### Key Libraries Used
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, catboost
- **Visualization:** matplotlib, seaborn
- **Model Evaluation:** StratifiedKFold, cross_validate

### Reproducibility
- Random seed: 42 for all models
- 5-fold stratified cross-validation
- Consistent preprocessing pipeline

## 📊 Results & Insights

### Model Performance Analysis
The CatBoost model significantly outperformed baseline models, achieving an AUC of 0.8565 in cross-validation and 0.87141 on the test set. Key factors contributing to success:

1. **Class Imbalance Handling:** Used class weights to address the severe imbalance (4.87% positive cases)
2. **Categorical Feature Optimization:** Proper handling of categorical variables without information loss
3. **Feature Engineering:** Created BMI missing indicator to capture missing data patterns
4. **Hyperparameter Tuning:** Optimized learning rate, depth, and iterations

### Feature Importance
The model identified key risk factors for stroke prediction:
- Age (strongest predictor)
- Average glucose level
- Hypertension
- Heart disease
- BMI

## 🚀 Usage

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Stroke-Risk-Prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook "657_Assignment4 (1).ipynb"
   ```

4. **Generate predictions:**
   - The notebook includes the complete pipeline
   - Final model generates `submission.csv` for Kaggle submission

## 📁 Project Structure

```
Stroke-Risk-Prediction/
├── 657_Assignment4 (1).ipynb    # Main analysis notebook
├── README.md                     # This file
├── train.csv                     # Training data (not included)
├── test.csv                      # Test data (not included)
└── submission.csv                # Generated predictions
```

## 🔬 Methodology Highlights

### Cross-Validation Strategy
- **5-fold Stratified Cross-Validation:** Ensures representative performance estimates
- **AUC as Primary Metric:** Appropriate for imbalanced classification
- **Early Stopping:** Prevents overfitting in gradient boosting

### Preprocessing Pipeline
- **Robust Imputation:** Median for numerical, constant for categorical
- **Feature Scaling:** StandardScaler for distance-based algorithms
- **Categorical Encoding:** One-hot encoding with first category drop

## 🎯 Future Improvements

1. **Ensemble Methods:** Stacking with XGBoost and LightGBM
2. **Feature Engineering:** Domain-specific feature creation
3. **Hyperparameter Optimization:** Bayesian optimization for better tuning
4. **Interpretability:** SHAP analysis for feature importance
5. **Model Deployment:** API development for real-time predictions

## 📚 References

- [CatBoost Documentation](https://catboost.ai/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/)
- [Kaggle Competition Page](https://www.kaggle.com/competitions/introtoml)

## 👨‍💻 Author

This project was developed as part of the Introduction to Machine Learning course (ECE 457B/657) at the University of Waterloo. The implementation demonstrates a complete end-to-end machine learning pipeline from data exploration to final model deployment, addressing real-world healthcare challenges through advanced machine learning techniques.

## 📄 Citation

Amir-Hossein Karimi and Arezoo Alipanah. IntroToML(ECE 457B|657) Kaggle Competition. https://kaggle.com/competitions/introtoml, 2025. Kaggle.

---

**Note:** This project achieved **10th place out of 175 submissions** in the Kaggle competition, demonstrating strong performance in stroke risk prediction using advanced machine learning techniques.