from sklearn import datasets

def load_dataset(dataset_name):
    if dataset_name == 'iris':
        data = datasets.load_iris()
    elif dataset_name == 'digits':
        data = datasets.load_digits()
    elif dataset_name == 'wine':
        data = datasets.load_wine()
    else:
        raise ValueError("Unknown dataset")

    return data.data, data.target



import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_iris(algorithm):
    data = datasets.load_iris()
    X, y = data.data, data.target
    
    if algorithm in ['linear_regression', 'ridge', 'lasso']:
        # Use 'petal length' to predict 'petal width'
        y = data.data[:, 3]  # 'petal width'
        X = data.data[:, 2].reshape(-1, 1)  # 'petal length'

    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y


def preprocess_digits(algorithm):
    data = datasets.load_digits()
    X, y = data.data, data.target

    if algorithm in ['linear_regression', 'ridge', 'lasso']:
        # Predict one pixel value based on another
        # For example, use pixel 21 to predict pixel 20
        y = data.data[:, 20]  # Pixel 20
        X = data.data[:, 21].reshape(-1, 1)  # Pixel 21

    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y


def preprocess_wine(algorithm):
    data = datasets.load_wine()
    X, y = data.data, data.target

    if algorithm in ['linear_regression', 'ridge', 'lasso']:
        # Predict 'Alcohol' content (feature 0) based on 'Color intensity' (feature 9)
        y = data.data[:, 0]  # 'Alcohol'
        X = data.data[:, 9].reshape(-1, 1)  # 'Color intensity'

    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y

