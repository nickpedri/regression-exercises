from env import get_connection
from sklearn.model_selection import train_test_split
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


def wrangle_zillow():
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
    zil.fips = zil.fips.astype(int)  # Convert fips to integer since it is not a decimal
    zil.bedrooms = zil.bedrooms.astype(int)  # Converts bedroom to integer since it is not a decimal
    return zil  # Return zil dataframe


'''This function will both retrieve AND clean the zillow data set. It used the zillow_data function to retrieve the
zillow data then drops all of the rows with null values. After dropping nulls, it creates a dictionary to rename the
zillow dataframe columns to something smaller and more concise. Then, it returns the clean dataframe.'''


def train_val_test(df, strat='None', seed=100, stratify=False):
    if stratify:
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed, stratify=df[strat])
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed, stratify=val_test[strat])
        return train, val, test
    if not stratify:
        train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
        val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
        return train, val, test


'''This function takes in a dataframe and splits the data into 3 separate dataframes containing 70%, 15% and 15% of 
the original data. It is used to split our data into a train, test, and validate sample. It has an argument for
stratify so you can choose if you want to stratify or not.'''
