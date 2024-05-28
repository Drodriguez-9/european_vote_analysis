import pandas as pd

def calculate_columns(df: pd.DataFrame):
    # Calculate the total amount of interest
    df['_TotalInterest'] = df['LoanAmount'] * df['InterestRate'] * df['LoanTerm']
    # Other derived columns
    df['_MonthlyPayment'] = df['LoanAmount'] / df['LoanTerm']
    df['_TotalPayment'] = df['LoanAmount'] + df['_TotalInterest']
    df['_LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']
    df['_RemainingIncomeAfterPayment'] = df['Income'] - df['_MonthlyPayment']
    df['_YearsEmployed'] = df['MonthsEmployed'] / 12
    df['_LoanTermYears'] = df['LoanTerm'] / 12
    df['_InterestToPrincipalRatio'] = df['_TotalInterest'] / df['LoanAmount']
    df["_NetIncome"]=df["Income"]*(1-df["DTIRatio"])
    return df