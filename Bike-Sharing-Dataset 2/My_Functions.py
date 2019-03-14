import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from collections import defaultdict

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score,roc_curve
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict,validation_curve
from sklearn.ensemble import RandomForestRegressor


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler,PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator , MultipleLocator
from gplearn.genetic import SymbolicRegressor

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression,LinearRegression, OrthogonalMatchingPursuit
from sklearn.model_selection import train_test_split , TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from matplotlib.gridspec import GridSpec
import plotly.tools as tls
import plotly
import plotly.plotly as py
from sklearn.decomposition import PCA
from pandas import DataFrame 
from sklearn.exceptions import ConvergenceWarning
from gplearn.genetic import SymbolicTransformer
from scipy.stats import *
from astral import Astral
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
import warnings
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score,roc_curve
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict,validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import SGDRegressor

plotly.tools.set_credentials_file(username='Furqan92', api_key='22DfVN5rFRg79OYygN5h')

tscv = TimeSeriesSplit(n_splits=5)
random_seed = 1234


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

## Genetic programming function that will create new features
def Genetic_P(dataset,target):
    append = 'mean_per_hour'
    a = dataset[append]
    y = dataset[target]
    X=dataset.copy()
    X=X.drop(target,axis=1)
    X=X.drop(append,axis =1)
    function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min','sin',
                 'cos',
                 'tan']
    gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=15,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=random_seed, n_jobs=3)
    gp_features = gp.fit_transform(X,y)
    print('Number of features created out of genetic programing: {}'.format(gp_features.shape))
    n = pd.DataFrame(gp_features)
    n =n.set_index(dataset.index.values)
    new_X = pd.concat([dataset, n],axis=1)
    new_X = new_X.dropna()
    return new_X

## Creating a new variable that compares the value to the past 7 days 
## the first 5 rows will be dropped if 'windspeed'is calculated and only 2 for the rest 
def relative_values(dataset, columns):
    dataset = dataset.copy()
    max = {'temp':41,'atemp':50,'hum':100,'windspeed':67}
    for i in columns:
        true=dataset[i]*max[i]
        avg7 = true.rolling(min_periods=1,window=24*7).mean().shift()
        std7 = true.rolling(min_periods=1,window=24*7).std().shift()
        name = 'relative_' + i 
        dataset[name]= (true - avg7)/std7
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    return dataset 
        
def check_skewness(df, numerical_cols, p_threshold=(0.75)):
    skewed_features = list()
    for feature in numerical_cols:
        data = df[feature].copy()
        skewness = skew(data)
        print("{} skewness p-value : {}".format(feature, skewness))

        if abs(skewness > p_threshold):
            print(feature)
            print("SKEWED")
            skewed_features.append(feature)
            print("-------------\n")
    print("\n------\n")
    print("skewed_features:")
    print(skewed_features)

    plt.rcParams["figure.figsize"] = (10,5)
    for i, feature in enumerate(skewed_features):
        plt.hist(df[feature], bins='auto')
        plt.title(feature)

# Preperation for isDaylight()
city_name = 'Washington DC'
a = Astral()
a.solar_depression = 'civil'
city = a[city_name]

def isDaylight(row):
    sun = city.sun(date=row['dteday'], local=True)
    row['isDaylight'] = 1 if (row['hr'] < sun['sunset'].hour and row['hr'] > sun['sunrise'].hour) else 0
    row['isNoon'] = 1 if row['hr'] == sun['noon'].hour else 0
    return row

def addRushHourFlags(row):
    #weekend
    if row['workingday'] == 0 :
        if row['hr'] in [10, 11, 12, 13, 14, 15, 16, 17, 18]:
            row['RushHour-High'] = 1
        elif row['hr'] in [8, 9, 19, 20, 21, 22, 23 ,0]:
            row['RushHour-Med'] = 1
        else:
            row['RushHour-Low'] = 1
    #weekdays
    if row['workingday'] == 1:
        if row['hr'] in [7, 8,9, 16, 17, 18, 19, 20]:
            row['RushHour-High'] = 1
        elif row['hr'] in [6,  10, 11, 12, 13, 15 ,21 ,22 ,23]:
            row['RushHour-Med'] = 1
        else:
            row['RushHour-Low'] = 1
    return row

def r2score(x,y):
    s = explained_variance_score(x,y)
    return s 


### This function will calculate the mean of the cnt of the previous 2 weeks during the same hour
def mean_per_hour_3weeks(dataset):
    a = [] 
    for i in range(0,len(dataset)):
        a.append(dataset[ (dataset['dteday']>= (dataset['dteday'].iloc[i] + datetime.timedelta(-21))) & ( dataset['dteday'] < (dataset['dteday'].iloc[i])) &( dataset['hr']  == dataset['hr'].iloc[i])]['cnt'].mean())
    dataset['mean_per_hour']= a
    dataset= dataset.dropna()
    return dataset

