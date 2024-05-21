import pandas as pd

def calculate_columns(df: pd.DataFrame):
    # Calculate the total amount of interest
    df['TotalInterest'] = df['LoanAmount'] * df['InterestRate'] * df['LoanTerm']
    # Other derived columns
    df['MonthlyPayment'] = df['LoanAmount'] / df['LoanTerm']
    df['TotalPayment'] = df['LoanAmount'] + df['TotalInterest']
    df['LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']
    df['RemainingIncomeAfterPayment'] = df['Income'] - df['MonthlyPayment']
    df['YearsEmployed'] = df['MonthsEmployed'] / 12
    df['LoanTermYears'] = df['LoanTerm'] / 12
    df['InterestToPrincipalRatio'] = df['TotalInterest'] / df['LoanAmount']
    return df