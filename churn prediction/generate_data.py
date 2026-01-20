"""
Synthetic Customer Churn Dataset Generator
Creates realistic customer data for churn prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_samples=5000, random_state=42):
    """
    Generate synthetic customer churn dataset
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Customer dataset with features and target variable
    """
    np.random.seed(random_state)
    
    # Basic demographics
    ages = np.random.normal(45, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    genders = np.random.choice(['Male', 'Female', 'Other'], n_samples)
    
    # Geographic data
    countries = np.random.choice(
        ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia'],
        n_samples,
        p=[0.4, 0.15, 0.15, 0.15, 0.10, 0.05]
    )
    
    # Account features
    tenure_months = np.random.exponential(20, n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)
    
    # Service usage
    monthly_charges = np.random.normal(65, 25, n_samples)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = tenure_months * monthly_charges + np.random.normal(0, 100, n_samples)
    total_charges = np.clip(total_charges, 0, None)
    
    # Contract and service features
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        n_samples,
        p=[0.55, 0.21, 0.24]
    )
    
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        n_samples,
        p=[0.34, 0.19, 0.22, 0.25]
    )
    
    # Service add-ons
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1])
    multiple_lines = np.random.choice(
        ['Yes', 'No', 'No phone service'],
        n_samples,
        p=[0.42, 0.48, 0.10]
    )
    
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        n_samples,
        p=[0.34, 0.44, 0.22]
    )
    
    online_security = np.random.choice(
        ['Yes', 'No', 'No internet service'],
        n_samples,
        p=[0.35, 0.50, 0.15]
    )
    
    online_backup = np.random.choice(
        ['Yes', 'No', 'No internet service'],
        n_samples,
        p=[0.35, 0.50, 0.15]
    )
    
    device_protection = np.random.choice(
        ['Yes', 'No', 'No internet service'],
        n_samples,
        p=[0.34, 0.51, 0.15]
    )
    
    tech_support = np.random.choice(
        ['Yes', 'No', 'No internet service'],
        n_samples,
        p=[0.29, 0.56, 0.15]
    )
    
    streaming_tv = np.random.choice(
        ['Yes', 'No', 'No internet service'],
        n_samples,
        p=[0.38, 0.47, 0.15]
    )
    
    streaming_movies = np.random.choice(
        ['Yes', 'No', 'No internet service'],
        n_samples,
        p=[0.39, 0.46, 0.15]
    )
    
    # Customer behavior features
    num_service_calls = np.random.poisson(1.5, n_samples)
    num_service_calls = np.clip(num_service_calls, 0, 10)
    
    days_since_last_interaction = np.random.exponential(30, n_samples).astype(int)
    days_since_last_interaction = np.clip(days_since_last_interaction, 0, 365)
    
    # Generate churn based on features (realistic relationships)
    churn_prob = np.zeros(n_samples)
    
    # Higher churn for month-to-month contracts
    churn_prob += (contract_types == 'Month-to-month') * 0.3
    churn_prob += (contract_types == 'One year') * 0.15
    churn_prob += (contract_types == 'Two year') * 0.05
    
    # Higher churn for higher charges
    churn_prob += (monthly_charges > 70) * 0.2
    churn_prob += (monthly_charges > 90) * 0.1
    
    # Higher churn for electronic check payment
    churn_prob += (payment_methods == 'Electronic check') * 0.15
    
    # Higher churn for shorter tenure
    churn_prob += (tenure_months < 12) * 0.25
    churn_prob += (tenure_months < 6) * 0.15
    
    # Higher churn for more service calls
    churn_prob += num_service_calls * 0.05
    churn_prob += (num_service_calls >= 4) * 0.2
    
    # Higher churn for no add-ons
    churn_prob += (online_security == 'No') * 0.1
    churn_prob += (tech_support == 'No') * 0.1
    
    # Add noise
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Convert to binary churn
    churn = (churn_prob > 0.35).astype(int)
    churn_labels = ['No' if c == 0 else 'Yes' for c in churn]
    
    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(1, n_samples + 1)],
        'age': ages,
        'gender': genders,
        'country': countries,
        'tenure_months': tenure_months,
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'num_service_calls': num_service_calls,
        'days_since_last_interaction': days_since_last_interaction,
        'churn': churn_labels
    })
    
    return data

if __name__ == "__main__":
    print("Generating synthetic customer churn dataset...")
    df = generate_customer_data(n_samples=5000)
    
    # Save to CSV
    df.to_csv('customer_churn_data.csv', index=False)
    print(f"\nDataset generated successfully!")
    print(f"Shape: {df.shape}")
    churn_counts = df['churn'].value_counts(normalize=True)
    churn_rate = churn_counts.get('Yes', 0) * 100
    print(f"Churn rate: {churn_rate:.2f}%")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset saved to 'customer_churn_data.csv'")
