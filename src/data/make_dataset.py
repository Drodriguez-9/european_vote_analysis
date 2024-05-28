import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import seaborn as sns
import torch
import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam
import itertools
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from pathlib import Path

PARENTS_PATH = Path(__file__).parents

def preprocess_data():
    data_path = PARENTS_PATH[2].joinpath("data", "raw", "data.csv")
    loan_default_df = pd.read_csv(data_path)
    categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 
                        'HasDependents', 'LoanPurpose', 'HasCoSigner']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}

    for col in categorical_columns:
        loan_default_df[col] = label_encoders[col].fit_transform(loan_default_df[col])

    # Normalize numerical features
    numerical_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    scaler = StandardScaler()
    loan_default_df[numerical_columns] = scaler.fit_transform(loan_default_df[numerical_columns])
    # Drop ID
    loan_default_df.drop(columns=['LoanID'], inplace=True)
    return loan_default_df

def split_data(df, test_size=0.2, random_state=42):
    target = 'Default'
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

        
def load_data(parent_dir):
    data_dir = parent_dir[2].joinpath("data", "processed", "processed_data.csv")
    df = pd.read_csv(data_dir)
    return df

def save_data(df: pd.DataFrame):
    data_path = PARENTS_PATH[2].joinpath("data", "processed")
    df.to_csv(data_path.joinpath("processed_data.csv"), index=False)


if __name__ == "__main__":
    loan_default_df = preprocess_data()
    save_data(loan_default_df)
    