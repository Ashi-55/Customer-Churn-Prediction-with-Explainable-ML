# Customer Churn Prediction with Explainable ML

A complete end-to-end machine learning project for predicting customer churn with comprehensive explainability using SHAP values.

## 📋 Overview

This project demonstrates a full ML pipeline from data generation to model interpretation, including:
- **Data Cleaning**: Handling missing values, data type conversions
- **Feature Engineering**: Creating meaningful business features
- **Model Training**: XGBoost classifier for churn prediction
- **Model Evaluation**: Accuracy, ROC-AUC, confusion matrix, feature importance
- **Explainability**: SHAP values for model interpretability
- **Business Insights**: Actionable recommendations based on model findings

## 🚀 Features

### Machine Learning Components
- ✅ Synthetic data generation for realistic customer datasets
- ✅ Exploratory data analysis (EDA)
- ✅ Data cleaning and preprocessing
- ✅ Advanced feature engineering (7+ derived features)
- ✅ Categorical encoding and scaling
- ✅ XGBoost model training with hyperparameters
- ✅ Comprehensive model evaluation metrics
- ✅ SHAP-based model explainability

### Evaluation Metrics
- **Accuracy**: Classification accuracy on train/test sets
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Visual representation of predictions
- **Feature Importance**: XGBoost and SHAP feature rankings
- **Precision-Recall Curve**: Additional evaluation metric

### Visualizations
- Churn distribution analysis
- Confusion matrix heatmap
- ROC curve
- Precision-Recall curve
- Feature importance plots
- SHAP summary plots
- SHAP waterfall plots

## 📦 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Step 1: Generate Data (Optional)
If you don't have a dataset, generate synthetic customer data:
```bash
python generate_data.py
```

This will create `customer_churn_data.csv` with 5,000 customer records.

### Step 2: Run the ML Pipeline
```bash
python churn_prediction.py
```

This will execute the complete pipeline:
1. Load and explore the data
2. Clean and preprocess
3. Engineer new features
4. Encode categorical variables
5. Split into train/test sets
6. Train XGBoost model
7. Evaluate model performance
8. Generate SHAP explanations
9. Provide business insights

## 📊 Output Files

The pipeline generates several visualization files:

1. **churn_distribution.png**: Distribution of churn vs no-churn customers
2. **model_evaluation.png**: Confusion matrix, ROC curve, and PR curve
3. **feature_importance.png**: Top 15 most important features
4. **shap_summary.png**: SHAP summary plot showing feature impacts
5. **shap_bar.png**: Mean absolute SHAP values by feature
6. **shap_waterfall.png**: Example prediction explanation

## 📈 Model Performance

The XGBoost model typically achieves:
- **Test Accuracy**: ~80-85%
- **ROC-AUC**: ~0.85-0.90
- **Good balance** between precision and recall

## 🔍 Feature Engineering

The pipeline creates several engineered features:
1. **num_services**: Count of subscribed services
2. **avg_monthly_charge**: Average charge per month
3. **charge_ratio**: Ratio of monthly to average charges
4. **tenure_group**: Categorical tenure groupings
5. **high_value_customer**: Flag for top 20% by total charges
6. **estimated_lifetime_value**: Customer lifetime value estimate
7. **interaction_frequency**: Frequency of customer interactions

## 💡 Business Insights

The model identifies key churn drivers:
- **Contract Type**: Month-to-month contracts have highest churn
- **Tenure**: Shorter tenure customers are more likely to churn
- **Service Calls**: More service calls correlate with higher churn
- **Payment Method**: Electronic check users have higher churn
- **Monthly Charges**: Higher charges increase churn probability

### Recommendations:
1. Focus retention efforts on month-to-month contract customers
2. Offer incentives for longer-term contracts
3. Improve service quality to reduce service calls
4. Promote online security and tech support subscriptions
5. Target customers with <12 months tenure

## 🛠️ Technology Stack

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: ML preprocessing and evaluation metrics
- **XGBoost**: Gradient boosting classifier
- **SHAP**: Model explainability
- **Matplotlib & Seaborn**: Data visualization

## 📝 Project Structure

```
.
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── generate_data.py           # Synthetic data generation
├── churn_prediction.py        # Main ML pipeline
├── customer_churn_data.csv    # Dataset (generated)
└── *.png                      # Generated visualizations
```

## 🔬 Model Architecture

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Objective**: Binary classification (churn vs no churn)
- **Evaluation Metric**: AUC (Area Under Curve)
- **Hyperparameters**:
  - Max Depth: 6
  - Learning Rate: 0.1
  - N Estimators: 100
  - Subsample: 0.8
  - Colsample by Tree: 0.8

## 🎯 Explainability

SHAP (SHapley Additive exPlanations) provides:
- **Global Explainability**: Feature importance across all predictions
- **Local Explainability**: Individual prediction explanations
- **Feature Interactions**: How features contribute to predictions
- **Model Transparency**: Understandable decision-making process

## 📚 Key Concepts Demonstrated

1. **Data Cleaning**: Handling missing values, type conversions
2. **Feature Engineering**: Creating domain-relevant features
3. **Model Training**: Training and tuning XGBoost
4. **Model Evaluation**: Multiple metrics and visualizations
5. **Explainable AI**: SHAP for model interpretability
6. **Business Relevance**: Translating ML results to actionable insights

## 🚦 Getting Started Example

```python
from churn_prediction import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor(random_state=42)

# Load data
predictor.load_data('customer_churn_data.csv')

# Run complete pipeline
predictor.explore_data()
predictor.clean_data()
predictor.engineer_features()
predictor.encode_features()
predictor.prepare_data()
predictor.train_model()
results = predictor.evaluate_model()
predictor.explain_model()
predictor.business_insights(results['feature_importance'])
```

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and use this project for learning ML best practices!

---

**Built with ❤️ for Machine Learning Education**
