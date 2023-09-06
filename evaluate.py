# import pandas as pd
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_residuals(df, actual, model):
    df['residuals'] = df[model] - df[actual]
    sns.scatterplot(data=df, x=actual, y='residuals')
    plt.hlines(0, 0, color='firebrick')


def regression_errors(data, actual='', model=''):
    df = data.copy()
    df['m_res'] = df[model] - df[actual]
    SSE = (df['m_res']**2).sum()
    ESS = sum((df[model] - df[actual].mean())**2)
    TSS = SSE + ESS
    MSE = SSE/len(data)
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE


def baseline(data, actual='', method='mean'):
    df = data.copy()
    if method == 'mean':
        df['baseline'] = df[actual].mean()
    elif method == 'median':
        df['baseline'] = df[actual].median()
    df['b_res'] = df['baseline'] - df[actual]
    SSE = (df['b_res'] ** 2).sum()
    MSE = SSE / len(data)
    RMSE = sqrt(MSE)
    return SSE, MSE, RMSE


def compare_model_base(data, actual='', model='', baseline_model=''):
    df = data.copy()
    df['b_res'] = df[baseline_model] - df[actual]
    df['m_res'] = df[model] - df[actual]
    SSE_baseline = (df['b_res']**2).sum()
    SSE = (df['m_res']**2).sum()
    print(f'The model SSE was {SSE}.')
    print(f'The baseline SSE was {SSE_baseline}.')
    if SSE < SSE_baseline:
        print(f'The model did better.')
    else:
        print(f'The baseline did better.')


def eval_model(y_actual, y_hat):
    return sqrt(mean_squared_error(y_actual, y_hat))


def train_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)

    train_rmse = eval_model(y_train, train_preds)

    val_preds = model.predict(X_val)

    val_rmse = eval_model(y_val, val_preds)

    print(f'The train RMSE is {train_rmse}.')
    print(f'The validate RMSE is {val_rmse}.')

    return model
