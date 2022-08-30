# how many missing values exist or better still what is the % of missing values in the dataset?
import numpy as np
import pandas as pd
import seaborn as sns


def convert_labels(df):
    df.columns = [column.replace(' ', '_').lower() for column in df.columns]
    return df


def percent_missing(df: pd.DataFrame):
    # Calculate total  number of cells in dataframe
    totalCells = np.product(df.shape)

    # Count number of missing values per column
    missingCount = df.isnull().sum()

    # Calculate total number of missing values
    totalMissing = missingCount.sum()

    # Calculate percentage of missing values
    return print("The dataset contains", round(((totalMissing / totalCells) * 100), 2), "%", "missing values.")


