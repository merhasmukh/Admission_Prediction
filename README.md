# University Admission + Dropout Prediction Model

A comprehensive machine learning system to predict university admission outcomes and weekly dropout patterns.

## 🎯 Project Goals

This system predicts three key outcomes:
1. **Admission Confirmation**: Whether a student will confirm their admission (`is_confirmed`)
2. **Joining Prediction**: Whether a confirmed student will actually join (`is_joined`)
3. **Weekly Dropout Tracking**: Whether a joined student will drop out in weeks 1-4 (`weekly_dropout`)

## 📊 Dataset Features

| Feature | Description | Example |
|---------|-------------|----------|
| Application ID | Unique identifier | 001 |
| Gender | Male/Female | Male |
| 12th Marks (%) | Class 12 percentage | 78 |
| Entrance Exam Score | Entrance test score | 56 |
| Form Submission Date | Application date | 2025-06-01 |
| Admission Confirmed | Confirmation status | 1 (Yes) |
| Joined | Joining status | 1 (Yes) |
| Week1_Dropout | Week 1 dropout | 0 |
| Week2_Dropout | Week 2 dropout | 0 |
| Week3_Dropout | Week 3 dropout | 1 |
| Week4_Dropout | Week 4 dropout | 0 |
| Final_Dropout_Label | Overall dropout | 1 (Yes) |

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Sample Data**:
   ```bash
   python data_generator.py
   ```

3. **Train Models**:
   ```bash
   python train_models.py
   ```

4. **Run Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

## 📁 Project Structure

```
Admission_Prediction/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_data.csv
├── models/
│   ├── admission_model.pkl
│   ├── joining_model.pkl
│   └── dropout_model.pkl
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
├── dashboard.py
├── api.py
├── train_models.py
├── data_generator.py
└── requirements.txt
```

## 🔧 Implementation Phases

- ✅ **Phase 1**: Data Collection & Design
- ✅ **Phase 2**: Data Preprocessing
- ✅ **Phase 3**: Model Building (3 models)
- ✅ **Phase 4**: Evaluation Metrics
- ✅ **Phase 5**: Weekly Dropout Dashboard
- ✅ **Phase 6**: Deployment Options
- 🔄 **Phase 7**: Monitoring & Improvement

## 📈 Models

1. **Admission Confirmation Model**: Predicts if student will confirm admission
2. **Joining Prediction Model**: Predicts if confirmed student will join
3. **Dropout Prediction Model**: Predicts weekly dropout patterns

## 🎛️ Deployment Options

- **Jupyter Notebook**: For internal analysis
- **Streamlit Dashboard**: Interactive web interface
- **Flask API**: REST endpoints for integration

## 📊 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: False positive control
- **Recall**: Actual dropout detection rate
- **F1-Score**: Balance of precision and recall

## 🔄 Future Enhancements

- Real-time prediction updates
- Advanced feature engineering
- Deep learning models
- Integration with university systems