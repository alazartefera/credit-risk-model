import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import os

# Load raw data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'data.csv')
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

# RFM Transformer
class RFMTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        snapshot_date = X['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = X.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).rename(columns={
            'TransactionStartTime': 'Recency',
            'TransactionId': 'Frequency',
            'Value': 'Monetary'
        })
        rfm.reset_index(inplace=True)
        return rfm

# Preprocessing pipeline
def preprocess_pipeline():
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    numeric_cols = ['Recency', 'Frequency', 'Monetary']

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

    return preprocessor

# Main function
def main():
    df = load_data()
    rfm = RFMTransformer().fit_transform(df)

    # Merge product/category info by CustomerId
    latest_info = df.groupby('CustomerId').last().reset_index()
    df_model = rfm.merge(latest_info[['CustomerId', 'ProductCategory', 'ChannelId', 'PricingStrategy']], on='CustomerId', how='left')

    # Create fraud labels: 1 if customer ever committed fraud
    fraud_labels = df.groupby("CustomerId")["FraudResult"].max().reset_index()
    df_model = df_model.merge(fraud_labels, on="CustomerId", how="left")

    # Preprocess features
    pipe = preprocess_pipeline()
    features = pipe.fit_transform(df_model.drop(columns=["FraudResult"]))
    target = df_model["FraudResult"]

    # Create output directory if not exists
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # Save processed features and target
    X_path = os.path.join(output_dir, "X.npy")
    y_path = os.path.join(output_dir, "y.npy")
    np.save(X_path, features.toarray() if hasattr(features, "toarray") else features)
    np.save(y_path, target.to_numpy())

    print(f"✅ Features saved to {X_path}")
    print(f"✅ Target saved to {y_path}")

if __name__ == "__main__":
    main()
