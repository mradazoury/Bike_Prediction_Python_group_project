import pandas as pd
import numpy as np
from scipy.stats import *

## Reading data 
def read_data(input_path):
    raw_data = pd.read_csv(input_path, keep_default_na=True)
    return raw_data


## Finding the correlation matrix for numerical variables
def correlation_spear(df):
    numeric_dtypes = ['int16', 'int32', 'int64',
                          'float16', 'float32', 'float64']
    numeric_features = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numeric_features.append(i)
    corr= stats.spearmanr(df[numeric_features])
    return pd.DataFrame(corr[0], columns=numeric_features,index= numeric_features)

### Dummifying categorical variables
def onehot_encode(df,category):
    df = df.copy()
    numericals = df.get(list(set(df.columns) - set(category)))
    new_df = numericals.copy()
    for categorical_column in category:
        new_df = pd.concat([new_df, 
                            pd.get_dummies(df[categorical_column], 
                                           prefix=categorical_column[0])], 
                           axis=1)
    return new_df

## Replacing number in season by real names and in weathersit by description
def num_name(df):
    df = df.copy()
    season = {2:'spring', 3:'summer', 4:'fall', 1:'winter'}
    df['season']= df.season.apply(
               lambda x: season[x]).astype('category') 
    weathersit = {1:'Good', 2:'Acceptable', 3:'Bad', 4:'Chaos'}
    df['weathersit']= df.weathersit.apply(
               lambda x: weathersit[x]).astype('category') 
    return df

## fixing desired types
def fix_types(df):
    df = df.copy()
    boolean = ['workingday','weekday','holiday']
    for j in boolean:
        df[j]= df[j].astype('int')
    return df