from env import get_connection
import pandas as pd
import os


def zillow_data():
    filename = 'zillow.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, 
                   taxamount, fips FROM properties_2017 WHERE propertylandusetypeid = 261'''
        url = get_connection('zillow')
        df = pd.read_sql(query, url)
        return df
