# Name: Isabella Livingston
# Description: Preprocessing for Kaggle Dataset

# Importing Assets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

# Main Function
def main():
    train_data = 'train.csv'
    test_data = 'test.csv'

    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    # Sets String Values (in Dataset) to numerical values
    for col in df_train.columns:
        if df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category':
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col])


    # Correlation Matrix
    corr_matrix = df_train.corr()
    target_corr = corr_matrix['is_claim'].sort_values(ascending=False)

    #print("Most correlated features (for is_claim):")
    #print(target_corr[1:11])

    #print("Least correlated features (for is_claim):")
    #print(target_corr[-10:])



    # Dropping these columns since they're the leat correlated features (makes the model more accurate)
    df_train_clean = df_train.drop(['policy_id', 'gear_box', 'transmission_type', 'rear_brakes_type',
                  'is_parking_camera', 'height', 'steering_type', 'max_torque',
                  'population_density', 'age_of_car'], axis=1)

    df_test_clean = df_test.drop(['policy_id', 'gear_box', 'transmission_type', 'rear_brakes_type',
                                  'is_parking_camera', 'height', 'steering_type', 'max_torque',
                                  'population_density', 'age_of_car'], axis=1)

    # Ready to use Training data
    X_train = df_train_clean.drop('is_claim', axis=1).values
    y_train = df_train_clean['is_claim'].values


main()
