import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def explore_data(data):
    print("Data Overview:")
    print(data.head())
    print("\nData Info:")
    print(data.info())
    print("\nData Summary Statistics:")
    print(data.describe())
    print("\nData Shape:")
    print(data.shape)

def handle_missing_data(data):
    missing_data = data.isna().sum()
    return data

def visualize_data(data):
    sns.pairplot(data)
    plt.show()

def statistical_analysis(data):
    correlation_matrix = data.corr()
    groupby_data = data.groupby('category_column').mean()
    return correlation_matrix, groupby_data

def data_cleaning(data):
    data.drop_duplicates(inplace=True)
    return data

def main():
    file_path = "../input/rawdata/ENSO.csv"

    data = load_data(file_path)

    explore_data(data)

    data = handle_missing_data(data)

    visualize_data(data)

    correlation_matrix, groupby_data = statistical_analysis(data)

    data = data_cleaning(data)

    print("EDA process completed.")

if __name__ == "__main__":
    main()