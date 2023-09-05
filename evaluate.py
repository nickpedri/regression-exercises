# import pandas as pd
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt


def plot_residuals(df, actual, model, hline=1_000):

    df['residuals'] = df[model] - df[actual]
    sns.scatterplot(data=df, x=actual, y='residuals')
    plt.hlines(0, 0, hline, color='firebrick')


def regression_errors(data, actual='', model=''):
    df = data.copy()
    df['m_res'] = df[model] - df[actual]
    SSE = (df['m_res']**2).sum()
    ESS = sum((df[model] - df[actual].mean())**2)
    TSS = SSE + ESS
    MSE = SSE/len(data)
    RMSE = sqrt(MSE)
    return SSE, ESS, TSS, MSE, RMSE


def baseline(data, actual, method='mean'):
    df = data.copy()
    if method == 'mean':
        df['baseline'] = df[actual].mean()
    if method == 'mode':
        pass
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
        print('The model did better.')
    else:
        print('The baseline did better.')
