from base64 import encode
import pandas as pd
import acquire
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def visualize_scaled_data(train, scaled_train):

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(scaled_train, bins=25, ec='black')
    plt.title('Scaled')

def fit_and_scale(scaler, train, validate, test):
    # only scales float columns
    orig = train.select_dtypes(include=np.number).columns
    floats = []
    id_logerror = ['logerror', 'parcelid']

    for col in orig:
        #print(col)
        if col not in id_logerror:
            floats.append(col)
    

    # fits scaler to training data only, then transforms 
    # train, validate & test
    scaler.fit(train[floats])
    scaled_train = pd.DataFrame(data=scaler.transform(train[floats]), columns=floats,)
    scaled_validate = pd.DataFrame(data=scaler.transform(validate[floats]), columns=floats)
    scaled_test = pd.DataFrame(data=scaler.transform(test[floats]), columns=floats)

    scaled_train = pd.concat([scaled_train.reset_index(), train[id_logerror].reset_index()], axis=1)
    scaled_validate = pd.concat([scaled_validate.reset_index(), validate.reset_index()], axis=1)
    scaled_test = pd.concat([scaled_test.reset_index(), test[id_logerror].reset_index()], axis=1)

    return scaled_train, scaled_validate, scaled_test, scaler

def prep_iris(iris_df):
    iris_data = iris_df
    col_drop = ['species_id', 'measurement_id']
    iris_data = iris_data.drop(columns=col_drop)
    
    iris_data = iris_data.rename(columns={'species_name': 'species'})
    print(iris_data.head())
    
    dummy_df = pd.get_dummies(iris_data['species'], dummy_na=False, drop_first=True)
    iris_data = pd.concat([iris_data, dummy_df], axis=1)
    
    return iris_data
    
def acquire_prepare_iris():
    iris_data = acquire.get_iris_data()
    iris_data = prep_iris(iris_data)
    return iris_data


def prep_titanic(titanic_df):
    
    titanic_dummy = pd.get_dummies(titanic_df[['embarked', 'sex', ]], dummy_na=False, drop_first=[True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy], axis=1)
    col_drop = ['class', 'deck', 'Unnamed: 0', 'embark_town', 'passenger_id','sex','embarked']
    titanic_df = titanic_df.drop(columns=col_drop)
    #print(titanic_df)

    categories = []
    quant_cols = []
    for col in titanic_df.columns:
        if titanic_df[col].nunique() < 10:
            categories.append(col)
        else:
            quant_cols.append(col)
    print(quant_cols)
    return titanic_df, categories, quant_cols

def acquire_prep_titanic():
    titanic_df = acquire.get_titanic_data()
    titanic_df, categories, quant_cols = prep_titanic(titanic_df)

    return titanic_df, categories, quant_cols

def contains_yes_no(df):
    categories_to_map = []
    for col in df.columns:
        if 'Yes' in df[col].unique():
            if 'No' in df[col].unique():
                if len(df[col].unique()) <= 3:
                    categories_to_map.append(col)
                
    return categories_to_map

def map_yes_nos(df):
    categories_to_map = contains_yes_no(df)
    
    for col in categories_to_map:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    return df


def prep_telco(df):
    
    unencoded_df = df.copy()
    """
    Takes in Telco_Churn Dataframe.
    Arguments: drops unnecessary columns, converts categorical data.
    Returns cleaned data.
    """
    #drop unneeded columns
    df.drop(columns=['internet_service_type_id',
                 'payment_type_id', 'contract_type_id', 'customer_id'], inplace=True)
    unencoded_df.drop(columns=['internet_service_type_id',
                 'payment_type_id', 'contract_type_id'], inplace=True)
    #drop null values stored as whitespace:
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != ""]
    unencoded_df['total_charges'] = unencoded_df['total_charges'].str.strip()
    unencoded_df = unencoded_df[unencoded_df.total_charges != ""]
    #convert to correct data type:
    df['total_charges'] = df.total_charges.astype(float)
    unencoded_df['total_charges'] = unencoded_df.total_charges.astype(float)
    #Convert binary categorical to numeric
    df = map_yes_nos(df)
    df = df.rename(columns={'gender': 'is_female'})
    df.is_female = df.is_female.map({'Female': 1, 'Male': 0})
    #Turn NaNs to 'None' 
    #All NaNs result from lacking a service
    # ex: online_sec: None because they have no
    # internet service. 
    # ex2: multiple_lines: None because they have
    # no phone service.
    for col in df.columns:
        if df[col].isna().sum() > 0:
           # print(col)
            df[col] = df[col].astype('object')
            df[col] = df[col].fillna(0)
    
    for col in unencoded_df.columns:
        if unencoded_df[col].isna().sum() > 0:
            #print(col)
            unencoded_df[col] = unencoded_df[col].astype('object')
            unencoded_df[col] = unencoded_df[col].fillna(0)

    #Turning all quantitative dtypes to float64
    #So I can loop and get them in a list
    df.tenure = df.tenure.astype('float64')
    df = df[df.tenure>0]

    quant_cols = []
    categories= []

    df = encode_object_columns(df)

    for col in df.columns:
        if len(df[col].unique()) < 5:
            categories.append(col)
        elif df[col].dtype == 'float64':
            quant_cols.append(col)

    #Get dummies for non-binary categorical variables:
    # dummy_df = pd.get_dummies(df[['contract_type', 'payment_type',
    #                           'internet_service_type']],
    #                           dummy_na = False,
    #                           drop_first=[True, True, True])
    #concatenate the two dataframes
    # df = pd.concat([df, dummy_df], axis=1)
    #df['customer_id'] = unencoded_df['customer_id']
    return df, categories, quant_cols, unencoded_df

def remove_outliers(threshold, quant_cols, df):
    z = np.abs((stats.zscore(df[quant_cols])))

    df_without_outliers=  df[(z < threshold).all(axis=1)]
    #print(df.shape)
    #print(df_without_outliers.shape)

    return df_without_outliers

def encode_train_validate_test(df, train, validate, test):
    
    train = encode_object_columns(train)
    validate = encode_object_columns(validate)
    test = encode_object_columns(test)

    train['customer_id'] = df['customer_id']
    validate['customer_id'] = df['customer_id']
    test['customer_id'] = df['customer_id']

    return train, validate, test

def encode_object_columns(train_df, drop_encoded=True):
    
    col_to_encode = object_columns_to_encode(train_df)
    dummy_df = pd.get_dummies(train_df[col_to_encode],
                              dummy_na=False,
                              drop_first=[True for col in col_to_encode])
    train_df = pd.concat([train_df, dummy_df], axis=1)
    train_df = train_df.drop(columns='Unnamed: 0')
    
    if drop_encoded:
        train_df = drop_encoded_columns(train_df, col_to_encode)

    return train_df

def object_columns_to_encode(train_df):
    object_type = []
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            object_type.append(col)

    return object_type

def drop_encoded_columns(train_df, col_to_encode):
    train_df = train_df.drop(columns=col_to_encode)
    return train_df

def acquire_prep_telco():
    telco_df = acquire.get_telco_data()
    telco_df, categories, quant_cols, unencoded_df = prep_telco(telco_df)

    return telco_df, categories, quant_cols, unencoded_df



if __name__ == '__main__':
    print(acquire_prepare_iris().head())
    print(acquire_prep_titanic()[0].head())
    print(acquire_prep_telco()[0].head())
