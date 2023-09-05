from env import get_connection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd
import os


def zillow_data():
    filename = 'zillow.csv'  # Checks for local file
    if os.path.isfile(filename):
        return pd.read_csv(filename)  # Returns local file if there is one
    else:
        query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, 
                   taxamount, fips FROM properties_2017 WHERE propertylandusetypeid = 261'''  # SQL query
        url = get_connection('zillow')  # Creates url to connect to server
        df = pd.read_sql(query, url)  # Queries data
        df.to_csv(filename, index=False)  # Saves data locally
        return df  # returns queried data


'''This function will acquire the data from the zillow database from CodeUp's database. The function checks if there is
 a local .csv for zillow and if there is then it reads and returns that file. If there isn't a local copy, then 
 the function will use a SQL query to retrieve the data and then save a local copy of zillow.csv.'''


def wrangle_zillow(new_fips=False):
    zil = zillow_data()  # Retriesves data and assigs it to variable within function
    zil = zil[zil.bedroomcnt.isna() != True]  # Drops bedroom count values that are null
    zil = zil[zil.taxvaluedollarcnt.isna() != True]  # Drops taxvaluedollarcnt values that are null
    zil = zil[zil.calculatedfinishedsquarefeet.isna() != True]  # Drops sq feet values that are null
    zil = zil[zil.yearbuilt.isna() != True]  # Drops year built values that are null
    zil = zil[zil.taxamount.isna() != True]  # Drops tax amount values that are null
    rename = {'bedroomcnt': 'bedrooms',  # Create a dictionary for new column names
              'bathroomcnt': 'bathrooms',
              'calculatedfinishedsquarefeet': 'sq_ft',
              'taxvaluedollarcnt': 'price',
              'yearbuilt': 'year_built',
              'taxamount': 'tax_amount'}
    zil = zil.rename(columns=rename)  # Rename colums using dictionary
    zil.year_built = zil.year_built.astype(int)
    zil.fips = zil.fips.astype(int)  # Convert fips to integer since it is not a decimal
    zil.bedrooms = zil.bedrooms.astype(int)  # Converts bedroom to integer since it is not a decimal
    if new_fips:
        fips_codes = {'6037': 'Los Angeles County',
                      '6059': 'Orange County',
                      '6111': 'Ventura County'}
        zil.fips = [fips_codes.get(fip) for fip in zil.fips.astype(str)]
    return zil  # Return zil dataframe


'''This function will both retrieve AND clean the zillow data set. It used the zillow_data function to retrieve the
zillow data then drops all of the rows with null values. After dropping nulls, it creates a dictionary to rename the
zillow dataframe columns to something smaller and more concise. Then, it returns the clean dataframe.'''


def train_val_test(df, strat='None', seed=100, stratify=False):  # Splits dataframe into train, val, test
    if stratify:  # Will split with stratify if stratify is True
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
        return train, val, test
    if not stratify:  # Will split without stratify if stratify is False
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
        return train, val, test


'''This function takes in a dataframe and splits the data into 3 separate dataframes containing 70%, 15% and 15% of 
the original data. It is used to split our data into a train, test, and validate sample. It has an argument for
stratify so you can choose if you want to stratify or not.'''


def scale_zillow(df='?', train=None, val=None, test=None, method='mms', scaled_cols=None):
    if train is None or val is None or test is None:
        train, val, test = train_val_test(df)
    if scaled_cols is None:
        scaled_cols = ['sq_ft', 'price', 'tax_amount']
    if method == 'mms':
        mms = MinMaxScaler()
        mms.fit(train[scaled_cols])
        train[scaled_cols] = mms.transform(train[scaled_cols])
        val[scaled_cols] = mms.transform(val[scaled_cols])
        test[scaled_cols] = mms.transform(test[scaled_cols])
        return train, val, test
    if method == 'ss':
        ss = StandardScaler()
        ss.fit(train[scaled_cols])
        train[scaled_cols] = ss.transform(train[scaled_cols])
        val[scaled_cols] = ss.transform(val[scaled_cols])
        test[scaled_cols] = ss.transform(test[scaled_cols])
        return train, val, test
    if method == 'rs':
        rs = RobustScaler()
        rs.fit(train[scaled_cols])
        train[scaled_cols] = rs.transform(train[scaled_cols])
        val[scaled_cols] = rs.transform(val[scaled_cols])
        test[scaled_cols] = rs.transform(test[scaled_cols])
        return train, val, test


'''This function scales zillow data. It takes in the zillow dataframe and splits it, or takes in the train, val, test
 dataframes. It accepts a string input for which method the data will be scaled by. It fits the data on the train
 dataframe then transforms all 3 dataframes.'''


def split_x_y(df, target=''):
    x_df = df.drop(columns=target)
    y_df = df[target]
    return x_df, y_df


def cheat_sheet():
    print(f'Residual is the difference between the observed value and predicted values.')
    print(f'SSE - Sum of the Squared Errors is the sum of the residuals squared.')
    print(f'MSE - Mean Squared Error is the SSE divided by the total number of data points (len).')
    print(f'RMSE - Root Mean Squared Error is the square root of the MSE.')
