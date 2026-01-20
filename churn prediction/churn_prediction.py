"""
Customer Churn Prediction with Explainable ML
End-to-end ML pipeline: data cleaning, feature engineering, model training, evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

class ChurnPredictor:
    """
    Complete ML pipeline for customer churn prediction
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_data(self, filepath):
        """Load the customer churn dataset"""
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        print(f"Data loaded: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Exploratory data analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nMissing Values:")
        print(self.df.isnull().sum()[self.df.isnull().sum() > 0])
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nChurn Distribution:")
        churn_counts = self.df['churn'].value_counts()
        print(churn_counts)
        churn_rate = churn_counts.get('Yes', 0) / len(self.df) * 100
        print(f"\nChurn Rate: {churn_rate:.2f}%")
        
        print(f"\nNumerical Features Summary:")
        print(self.df.select_dtypes(include=[np.number]).describe())
        
        # Visualize churn distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        churn_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Churn Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Churn', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        churn_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
        plt.title('Churn Rate', fontsize=14, fontweight='bold')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
        print("\nSaved: churn_distribution.png")
        
    def clean_data(self):
        """Data cleaning and preprocessing"""
        print("\n" + "="*60)
        print("DATA CLEANING")
        print("="*60)
        
        df_clean = self.df.copy()
        
        # Handle missing values
        if df_clean['total_charges'].dtype == 'object':
            # Replace empty strings with 0
            df_clean['total_charges'] = df_clean['total_charges'].replace(' ', '0')
            df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'])
        
        # Fill any remaining NaN values
        df_clean = df_clean.fillna(0)
        
        # Remove customer_id (not a feature)
        if 'customer_id' in df_clean.columns:
            df_clean = df_clean.drop('customer_id', axis=1)
        
        print(f"After cleaning: {df_clean.shape}")
        print(f"Missing values: {df_clean.isnull().sum().sum()}")
        
        self.df_clean = df_clean
        return df_clean
    
    def engineer_features(self):
        """Feature engineering"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        df_fe = self.df_clean.copy()
        
        # Create new features
        # 1. Service count (number of services subscribed)
        service_cols = ['phone_service', 'online_security', 'online_backup',
                       'device_protection', 'tech_support', 'streaming_tv',
                       'streaming_movies']
        
        df_fe['num_services'] = 0
        for col in service_cols:
            df_fe['num_services'] += (df_fe[col] == 'Yes').astype(int)
        
        # 2. Average charge per month (total_charges / tenure)
        df_fe['avg_monthly_charge'] = df_fe['total_charges'] / (df_fe['tenure_months'] + 1)
        
        # 3. Charge ratio (monthly_charges / avg_monthly_charge)
        df_fe['charge_ratio'] = df_fe['monthly_charges'] / (df_fe['avg_monthly_charge'] + 1)
        
        # 4. Tenure groups
        df_fe['tenure_group'] = pd.cut(
            df_fe['tenure_months'],
            bins=[0, 12, 24, 48, 100],
            labels=['Short', 'Medium', 'Long', 'Very Long']
        )
        
        # 5. High value customer (top 20% by total charges)
        df_fe['high_value_customer'] = (
            df_fe['total_charges'] > df_fe['total_charges'].quantile(0.8)
        ).astype(int)
        
        # 6. Customer lifetime value estimate
        df_fe['estimated_lifetime_value'] = df_fe['monthly_charges'] * df_fe['tenure_months']
        
        # 7. Interaction frequency (inverse of days since last interaction)
        df_fe['interaction_frequency'] = 365 / (df_fe['days_since_last_interaction'] + 1)
        
        print(f"Created {len([col for col in df_fe.columns if col not in self.df_clean.columns])} new features")
        print(f"Total features: {df_fe.shape[1] - 1}")  # -1 for target
        
        self.df_fe = df_fe
        return df_fe
    
    def encode_features(self):
        """Encode categorical features"""
        print("\n" + "="*60)
        print("FEATURE ENCODING")
        print("="*60)
        
        df_encoded = self.df_fe.copy()
        
        # Separate features and target
        target = df_encoded['churn']
        df_features = df_encoded.drop('churn', axis=1)
        
        # Convert categorical dtypes to object/string first
        for col in df_features.columns:
            if df_features[col].dtype.name == 'category':
                df_features[col] = df_features[col].astype(str)
        
        # Encode categorical variables
        categorical_cols = df_features.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_features[col] = le.fit_transform(df_features[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable
        target_le = LabelEncoder()
        target_encoded = target_le.fit_transform(target)
        self.label_encoders['churn'] = target_le
        
        # Store feature names
        self.feature_names = df_features.columns.tolist()
        
        print(f"Encoded {len(categorical_cols)} categorical features")
        print(f"Total features for model: {len(self.feature_names)}")
        
        self.X = df_features
        self.y = target_encoded
        
        return df_features, target_encoded
    
    def prepare_data(self, test_size=0.2):
        """Split data into train and test sets"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Train churn rate: {y_train.mean()*100:.2f}%")
        print(f"Test churn rate: {y_test.mean()*100:.2f}%")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, **xgb_params):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Default XGBoost parameters
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': 0
        }
        
        # Update with user-provided parameters
        default_params.update(xgb_params)
        
        print(f"Training XGBoost with parameters:")
        for key, value in default_params.items():
            print(f"  {key}: {value}")
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )
        
        print("Model trained successfully!")
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, y_test_proba)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test ROC-AUC: {test_auc:.4f}")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_test_pred, 
                                   target_names=['No Churn', 'Churn']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        plt.figure(figsize=(15, 5))
        
        # Confusion Matrix
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # ROC Curve
        plt.subplot(1, 3, 2)
        fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Precision-Recall Curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(self.y_test, y_test_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})', linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nSaved: model_evaluation.png")
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 15 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }
    
    def explain_model(self, n_samples=100):
        """Generate SHAP explanations for model predictions"""
        print("\n" + "="*60)
        print("MODEL EXPLAINABILITY (SHAP)")
        print("="*60)
        
        # Use subset for faster computation
        X_test_sample = self.X_test.sample(min(n_samples, len(self.X_test)), 
                                          random_state=self.random_state)
        
        print(f"Computing SHAP values for {len(X_test_sample)} samples...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, 
                         feature_names=self.feature_names, 
                         show=False, plot_size=(10, 8))
        plt.title('SHAP Summary Plot - Feature Impact on Churn', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        print("Saved: shap_summary.png")
        
        # Bar plot (mean SHAP values)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample,
                         feature_names=self.feature_names,
                         plot_type='bar', show=False)
        plt.title('SHAP Feature Importance (Mean Absolute Impact)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
        print("Saved: shap_bar.png")
        
        # Waterfall plot for a single prediction
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_test_sample.iloc[0].values,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title('SHAP Waterfall Plot - Example Prediction Explanation',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
        print("Saved: shap_waterfall.png")
        
        print("\nSHAP explanations generated successfully!")
        return explainer, shap_values
    
    def business_insights(self, feature_importance):
        """Generate business-relevant insights"""
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS")
        print("="*60)
        
        top_features = feature_importance.head(10)['feature'].tolist()
        
        print("\nKey Findings:")
        print("-" * 60)
        
        # Analyze top features
        for feature in top_features[:5]:
            if feature in self.df_fe.columns:
                churn_by_feature = self.df_fe.groupby(feature)['churn'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                )
                print(f"\n{feature.upper()}:")
                print(f"  Churn rates by category:")
                for cat, rate in churn_by_feature.items():
                    print(f"    {cat}: {rate:.2f}%")
        
        # Contract type analysis
        if 'contract_type' in self.df_fe.columns:
            contract_churn = self.df_fe.groupby('contract_type')['churn'].apply(
                lambda x: (x == 'Yes').mean() * 100
            )
            print("\nCONTRACT TYPE IMPACT:")
            for contract, rate in contract_churn.items():
                print(f"  {contract}: {rate:.2f}% churn rate")
        
        # Tenure analysis
        if 'tenure_months' in self.df_fe.columns:
            tenure_churn = self.df_fe.groupby(
                pd.cut(self.df_fe['tenure_months'], 
                      bins=[0, 6, 12, 24, 100],
                      labels=['0-6 months', '6-12 months', '12-24 months', '24+ months'])
            )['churn'].apply(lambda x: (x == 'Yes').mean() * 100)
            
            print("\nTENURE IMPACT:")
            for tenure, rate in tenure_churn.items():
                print(f"  {tenure}: {rate:.2f}% churn rate")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS:")
        print("="*60)
        print("1. Focus retention efforts on customers with:")
        print("   - Month-to-month contracts")
        print("   - Short tenure (<12 months)")
        print("   - High number of service calls")
        print("   - Electronic check payment method")
        print("2. Promote long-term contracts with incentives")
        print("3. Improve service quality to reduce service calls")
        print("4. Encourage online security and tech support subscriptions")


def main():
    """Main execution function"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION WITH EXPLAINABLE ML")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor(random_state=42)
    
    # Load data (generate if doesn't exist)
    try:
        predictor.load_data('customer_churn_data.csv')
    except FileNotFoundError:
        print("\nDataset not found. Generating synthetic data...")
        import generate_data
        generate_data.generate_customer_data(n_samples=5000, random_state=42)
        predictor.load_data('customer_churn_data.csv')
    
    # Run complete pipeline
    predictor.explore_data()
    predictor.clean_data()
    predictor.engineer_features()
    predictor.encode_features()
    predictor.prepare_data()
    predictor.train_model()
    evaluation_results = predictor.evaluate_model()
    predictor.explain_model(n_samples=100)
    predictor.business_insights(evaluation_results['feature_importance'])
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated Files:")
    print("  - churn_distribution.png")
    print("  - model_evaluation.png")
    print("  - feature_importance.png")
    print("  - shap_summary.png")
    print("  - shap_bar.png")
    print("  - shap_waterfall.png")


if __name__ == "__main__":
    main()
