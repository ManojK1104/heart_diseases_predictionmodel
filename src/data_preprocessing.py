from collections import OrderedDict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def processing(df):
    df.drop_duplicates()

    ''' 
    data dosenot contains missing values
    '''
    # numerical and categorical col seprate
    categotical_column=df.select_dtypes(include=object)
    numerical_column=df.select_dtypes(exclude='object')

    # removed unwanted column
    df.drop(columns=['patient_id',],inplace=True)

    # calulated outliers present on data ot not
    stats=[]
    for i in df.select_dtypes(include=('int64','float64')):
        numerical_stats=OrderedDict({
        "feacture":i,
        "count":df[i].count(),
        "mean":df[i].mean(),
        "median":df[i].median(),
        "std":df[i].std(),
        "variance":df[i].var(),
        "skew":df[i].skew(),
        "kurt":df[i].kurt(),
        "Q1":df[i].quantile(0.25),
        "Q3":df[i].quantile(0.75),
        "IQR":df[i].quantile(0.75)-df[i].quantile(0.25),
        "UW":(df[i].quantile(0.25)-1.5*(df[i].quantile(0.75)-df[i].quantile(0.25))),
        "LW":(df[i].quantile(0.75)+1.5*(df[i].quantile(0.75)-df[i].quantile(0.25)))

        })
        stats.append(numerical_stats)
        report=pd.DataFrame(stats)

    # categorical to numerical conversion
    lc=LabelEncoder()
    for i in df.select_dtypes(include=object):
        df[i]=lc.fit_transform(df[i])

    #seprated target column
    X=df.drop(columns=['has_heart_disease'])
    y=df['has_heart_disease']

    #train and test split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

    # did target column equal
    from imblearn.over_sampling import SMOTE
    sc=SMOTE()
    X_train,y_train=sc.fit_resample(X_train,y_train)

    return X_train,X_test,y_train,y_test