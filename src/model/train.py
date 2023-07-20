# Import libraries

import argparse
import glob
import os

import mlflow
import mlflow.sklearn


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# define functions
def main(args):
    # TO DO: enable autologging
    with mlflow.start_run():
        mlflow.sklearn.autolog()  # Enable autologging

        # read data
        df = get_csvs_df(args.training_data)

        # split data
        X_train, X_test, y_train, y_test = split_data(df)

        # train model
        model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

        evaluate_model(model, X_test, y_test)

    def get_csvs_df(path):
        if not os.path.exists(path):
            raise RuntimeError(f"Cannot use non-existent path provided: {path}")
        csv_files = glob.glob(f"{path}/*.csv")
        if not csv_files:
            raise RuntimeError(f"No CSV files found in provided data path: {path}")
        return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df):
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):    
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print(acc)

    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print(auc)

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
