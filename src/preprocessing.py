import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds = []
        self.upper_bounds = []
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds.append(q1 - self.factor * iqr)
            self.upper_bounds.append(q3 + self.factor * iqr)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        for i, col in enumerate(X.columns):
            X[col] = np.where(X[col] < self.lower_bounds[i], self.lower_bounds[i], X[col])
            X[col] = np.where(X[col] > self.upper_bounds[i], self.upper_bounds[i], X[col])
        return X.values

def preprocess_data():
    df = pd.read_csv('data/HIGGS.csv', header=None)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Outlier handling
    capper = OutlierCapper()
    X_capped = capper.fit_transform(X)
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_capped)
    
    return X_scaled, y