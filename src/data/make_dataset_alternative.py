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
    data_path = PARENTS_PATH[2].joinpath("data", "raw", "hmeq.csv")
    loan_default_df = pd.read_csv(data_path)
    #Replacing and dropping values
    replacement_values = {
    'DELINQ': 0,   
    'JOB': 'Missing',  
    'DEBTINC': loan_default_df['DEBTINC'].mean() 
    }

    loan_default_df.fillna(value=replacement_values, inplace=True)

    loan_default_df.dropna(inplace=True)

    categorical_columns = ['BAD','REASON','JOB']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}

    for col in categorical_columns:
        loan_default_df[col] = label_encoders[col].fit_transform(loan_default_df[col])

    # Normalize numerical features
    numerical_columns = ['LOAN','MORTDUE','VALUE','YOJ','DEROG','DELINQ','CLAGE','NINQ','CLNO','DEBTINC']
    scaler = StandardScaler()
    loan_default_df[numerical_columns] = scaler.fit_transform(loan_default_df[numerical_columns])
    return loan_default_df

def split_data(df, test_size=0.2, random_state=42):
    target = 'BAD'
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def save_data(df: pd.DataFrame):
    data_path = PARENTS_PATH[2].joinpath("data", "processed")
    df.to_csv(data_path.joinpath("processed_data.csv"), index=False)


if __name__ == "__main__":
    loan_default_df = preprocess_data()
    save_data(loan_default_df)



    