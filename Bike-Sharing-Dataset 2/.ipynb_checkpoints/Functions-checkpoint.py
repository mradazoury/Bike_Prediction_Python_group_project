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
