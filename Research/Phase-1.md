# Phase 1: Baseline Models

**Date Started:** 2025-07-25  
**Status:** In Progress  
**Objective:** Establish baseline performance for all three prediction tasks

## 🎯 Goals

1. Implement baseline models for:
   - Admission Confirmation Prediction (`is_confirmed`)
   - Joining Prediction (`is_joined`)
   - Weekly Dropout Prediction (`weekly_dropout`)

2. Establish performance benchmarks
3. Identify key features for each prediction task
4. Create reproducible training pipeline

## 📊 Dataset Overview

- **Total Samples:** 1000
- **Features:** 17 columns
- **Target Variables:** 3 (is_confirmed, is_joined, weekly_dropout)
- **Data Split:** 80% train, 20% test

### Feature Summary
| Feature | Type | Description | Missing Values |
|---------|------|-------------|----------------|
| Application_ID | Categorical | Unique identifier | 0% |
| Gender | Categorical | Male/Female | 0% |
| Marks_12th_Percent | Numerical | Class 12 percentage | 0% |
| Entrance_Exam_Score | Numerical | Entrance test score | 0% |
| Form_Submission_Date | Date | Application date | 0% |
| Is_Confirmed | Binary | Target 1 | 0% |
| Is_Joined | Binary | Target 2 | 0% |
| Week1-4_Dropout | Binary | Weekly dropout flags | 0% |
| Final_Dropout_Label | Binary | Target 3 | 0% |

## 🔧 Methodology

### Preprocessing Steps
1. **Date Features:** Extract day, month, year from submission date
2. **Categorical Encoding:** One-hot encoding for gender
3. **Feature Scaling:** StandardScaler for numerical features
4. **Train-Test Split:** Stratified split to maintain class balance

### Baseline Models

#### Model 1: Admission Confirmation
- **Algorithm:** Logistic Regression, Random Forest, XGBoost
- **Features:** Gender, Marks_12th_Percent, Entrance_Exam_Score, submission timing
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Model 2: Joining Prediction
- **Algorithm:** Logistic Regression, Random Forest, XGBoost
- **Features:** All features + is_confirmed
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

#### Model 3: Weekly Dropout Prediction
- **Algorithm:** Logistic Regression, Random Forest, XGBoost
- **Features:** All features + is_confirmed + is_joined
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📈 Experiments

### Experiment 1.1: Logistic Regression Baseline
**Date:** 2025-07-25  
**Status:** ⏳ Planned

**Configuration:**
```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
```

**Results:**
- Admission Confirmation: TBD
- Joining Prediction: TBD
- Dropout Prediction: TBD

### Experiment 1.2: Random Forest Baseline
**Date:** TBD  
**Status:** ⏳ Planned

**Configuration:**
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
```

**Results:**
- Admission Confirmation: TBD
- Joining Prediction: TBD
- Dropout Prediction: TBD

### Experiment 1.3: XGBoost Baseline
**Date:** TBD  
**Status:** ⏳ Planned

**Configuration:**
```python
XGBClassifier(
    random_state=42,
    eval_metric='logloss'
)
```

**Results:**
- Admission Confirmation: TBD
- Joining Prediction: TBD
- Dropout Prediction: TBD

## 📊 Results Summary

### Performance Comparison

| Model | Task | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|------|----------|-----------|--------|----------|----------|
| LogReg | Confirmation | - | - | - | - | - |
| LogReg | Joining | - | - | - | - | - |
| LogReg | Dropout | - | - | - | - | - |
| RF | Confirmation | - | - | - | - | - |
| RF | Joining | - | - | - | - | - |
| RF | Dropout | - | - | - | - | - |
| XGB | Confirmation | - | - | - | - | - |
| XGB | Joining | - | - | - | - | - |
| XGB | Dropout | - | - | - | - | - |

### Feature Importance Analysis

**Top Features for Admission Confirmation:**
1. TBD
2. TBD
3. TBD

**Top Features for Joining Prediction:**
1. TBD
2. TBD
3. TBD

**Top Features for Dropout Prediction:**
1. TBD
2. TBD
3. TBD

## 🔍 Analysis & Insights

### What Worked Well
- TBD

### Challenges Encountered
- TBD

### Key Observations
- TBD

### Data Quality Issues
- TBD

## 🚀 Next Steps (Phase 2)

1. **Feature Engineering:**
   - Create interaction features
   - Time-based features (seasonality, urgency)
   - Academic performance ratios

2. **Advanced Preprocessing:**
   - Handle class imbalance with SMOTE
   - Feature selection techniques
   - Polynomial features

3. **Model Improvements:**
   - Hyperparameter tuning
   - Cross-validation strategies
   - Ensemble methods

## 📝 Notes

- Remember to set random seeds for reproducibility
- Save all trained models in `models/phase1/` directory
- Document any data preprocessing decisions
- Consider computational time for larger datasets

## 🔗 Related Files

- `src/model_training.py` - Main training script
- `src/data_preprocessing.py` - Data preprocessing utilities
- `src/evaluation.py` - Model evaluation functions
- `data/sample_data.csv` - Training dataset