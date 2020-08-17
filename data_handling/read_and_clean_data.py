import pandas as pd
import re
from matplotlib import pyplot as plt
import numpy as np


def check_data(file_path) -> pd.DataFrame:
    """

    :param file_path:
    :return:
    """
    print("Check Raw Data")
    df = pd.read_csv(file_path)

    df.info()

    return df


def remove_features(df: pd.DataFrame) -> (pd.DataFrame, list):
    """
    Remove Features with missing values crossing the threshold
    :param df: Pandas DataFrame
    :return: Reduced Pandas DataFrame
    """
    print("Remove Features with too many missing values")
    threshold = 0.7
    data_df = df[df.columns[df.isnull().mean() < threshold]]

    data_df = data_df.loc[data_df.isnull().mean(axis=1) < threshold]

    removed_features = list((set(df.columns) - set(data_df.columns)))

    print("removed features: {}".format(removed_features))

    data_df.info()

    return data_df, removed_features


def correct_term_feature(value) -> float:
    """

    :param value:
    :return:
    """
    pattern = re.compile(r'([0-9]+)\s+month.+')

    search_results = pattern.search(value)

    if search_results:
        result = search_results.groups()[0]
        result = int(result)
    else:
        result = value
    return result


def correct_last_week_pay_feature(value) -> float:
    """

    :param value:
    :return:
    """

    pattern = re.compile(r'([0-9]+).+\sweek')
    searches = pattern.search(value)

    if searches:
        result = searches.groups()[0]
        result = int(result)
    else:
        result = value

    if result == 'NAth week':
        result = np.nan

    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean raw data and apply appropriate operations on the features
    :param df:
    :return:
    """
    print("Cleaning data")
    # correct term feature column and convert it into int/float object
    df['term'] = df['term'].apply(correct_term_feature).astype('float64')

    df['last_week_pay'] = df['last_week_pay'].apply(correct_last_week_pay_feature).astype('float64')

    # convert all the object type data features into category type data
    selected_data_types = df.select_dtypes(include=['object']).columns
    # print(df['emp_title'].value_counts())
    # convert the object data types column into category data type
    for i in selected_data_types:
        if i == 'member_id':
            pass
        df[i] = df[i].map(lambda x: x.lower() if isinstance(x, str) else x)
        # df[i] = df[i].astype('category').cat.codes
    df.info()

    # unique_batches = len(df['batch_enrolled'].unique())
    # print(unique_batches)

    # print(df.value_counts())
    return df


if __name__ == '__main__':
    dataset_file = 'dataset/train_indessa.csv'
    check_data(dataset_file)
