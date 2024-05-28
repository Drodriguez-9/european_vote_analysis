import pandas as pd

def calculate_columns(df: pd.DataFrame):
    # Calculate the total amount of interest
    #df['TotalInterest'] = df['LoanAmount'] * df['InterestRate'] * df['LoanTerm']
    # Other derived columns
    #df['MonthlyPayment'] = df['LoanAmount'] / df['LoanTerm']
    #df['TotalPayment'] = df['LoanAmount'] + df['TotalInterest']
    df['LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']
    df['YearsEmployed'] = df['MonthsEmployed'] / 12
    df['LoanTermYears'] = df['LoanTerm'] / 12

    # Includes monthly compounding
    df["TotalPayment"] = df["LoanAmount"] * (1 + df["InterestRate"]/100)**df["LoanTermYears"]
    df["TotalInterest"] = df["TotalPayment"] - df["LoanAmount"]
    df["MonthlyPayment"] = df["TotalPayment"]/df["LoanTerm"]
    df['RemainingIncomeAfterPayment'] = df['Income'] - df['MonthlyPayment']
    df['InterestToPrincipalRatio'] = df['TotalInterest'] / df['LoanAmount']


    return df