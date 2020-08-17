import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import mca


def get_training_data(df: pd.DataFrame) -> (np.ndarray, np.ndarray, dict, dict, list, list):
    """

    :param df:
    :return:
    """
    # first drop all the non-required columns/features
    cols_to_drop = ['member_id', 'title', 'emp_title', 'batch_enrolled']

    df.drop(labels=cols_to_drop, axis=1, inplace=True)

    target_variables = ['loan_status']

    # transform raw data to prepare for training
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    to_scaled_features = list((set(numerical_features) - set(['loan_status'])))

    df, median_dict, category_freq_value_dict = handle_missing_values(df, to_scaled_features, categorical_features)

    # scale features for handling the outliers and proper distribution of data
    df = scale_features(df, to_scaled_features)

    df, mean_enc_dicts = handle_categorical_features(df, categorical_features)



    plot_corr_mat(df)

    input_features = categorical_features + to_scaled_features
    print("input features: {}".format(input_features))
    input_features = df[input_features]

    print("final training df")
    input_features.info()
    training_cols = input_features.columns
    x = np.array(input_features)
    y = np.array(df[target_variables])

    return x, y, median_dict, category_freq_value_dict, mean_enc_dicts, to_scaled_features, categorical_features, training_cols


def mean_encoding(df, feature, target_feature) -> (list, dict):
    """
    Use mean encoding as Target encoding method for encoding the categorical features
    :param df:
    :param feature:
    :param target_feature:
    :return:
    """
    mean_encoded_dict = df.groupby([feature])[target_feature].mean().to_dict()
    df[feature] = df[feature].map(mean_encoded_dict)

    return df[feature].values, mean_encoded_dict


def scale_features(df: pd.DataFrame, features_list: list) -> pd.DataFrame:
    """
    scale features within a range to make the data distribution more regularised
    :param df:
    :param features_list:
    :return:
    """
    print("Scaling numerical features: {}".format(features_list))
    scaler = StandardScaler()

    scaler.fit(df[features_list])

    scaled_data = scaler.transform(df[features_list])

    with open('scaler.pkl', 'wb') as fobj:
        pickle.dump(scaler, fobj)
    # print(scaled_data)
    scaled_df = pd.DataFrame(scaled_data, columns=features_list)

    columns = df.columns

    other_columns = list((set(columns) - set(features_list)))

    non_scaled_df = df[other_columns]

    new_df = scaled_df.join(non_scaled_df)

    new_df.info()
    del scaled_df, non_scaled_df, df

    return new_df


def handle_categorical_features(df: pd.DataFrame, categorical_features: list) -> (pd.DataFrame, dict):
    """
    encode categorical features with their mean encoded values
    :param df:
    :param categorical_features:
    :return:
    """
    print("Handling categorical features: {}".format(categorical_features))
    # for converting categorical variables into columns of binary values
    # though this is only useful for categories with low cardinality
    # categorical_df = pd.get_dummies(df, prefix=categorical_features).astype(np.int8)
    #
    # dummies_columns = categorical_df.columns

    # for later use with test dataset apply the below
    # df.reindex(columns=dummies_columns, fill_value=0)

    # below method is more suitable for categorical variable with high cardinality
    # mca_df = mca.MCA(df, cols=categorical_features)
    # print(mca_df)
    # categorical_df.info()

    # mean encoding
    mean_encoding_dicts = dict()
    for i in categorical_features:
        df[i], enc_dict = mean_encoding(df, i, 'loan_status')
        mean_encoding_dicts[i] = enc_dict
    print(mean_encoding_dicts)
    df.info()

    return df, mean_encoding_dicts


def handle_missing_values(df: pd.DataFrame, numerical_features: list, categorical_features: list) -> (pd.DataFrame, dict, dict):
    """
    handle missing values and fill them using proper methods
    :param df:
    :param numerical_features:
    :param categorical_features:
    :return:
    """
    print("handling missing values")
    median_dict = dict()
    cat_fill_dict = dict()

    # for numerical features
    for i in numerical_features:
        median = df[i].median()
        df[i].fillna(value=median, inplace=True)
        median_dict[i] = median

    # for categorical Features fill it with most common value
    for i in categorical_features:
        frequent_value = df[i].value_counts().index[0]
        df[i].fillna(value=frequent_value, inplace=True)
        cat_fill_dict[i] = frequent_value

    df.info()
    return df, median_dict, cat_fill_dict


def plot_corr_mat(df: pd.DataFrame):
    """
    Plot correlation matrix, method: pearson
    :param df:
    :return:
    """
    # df.drop(labels=['member_id'], axis=1, inplace=True)
    # print(len(df['member_id'].unique()))
    correlation = df.corr()
    columns = df.columns
    # print(columns)
    columns_len = len(columns)
    # print(correlation)
    plt.matshow(correlation)
    plt.xticks(np.arange(columns_len), columns, rotation=90)
    plt.yticks(np.arange(columns_len), columns, rotation=0)
    plt.colorbar()
    plt.show()
    plt.savefig('coefs_mat.png')


def get_redundant_pairs(df):
    """
    Get diagonal and lower triangular pairs of correlation matrix
    :param df: data frame
    :return:
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    """
    Get top correlated features
    :param df: data frame
    :param n: top number of feature correlation to return
    :return:
    """
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


if __name__ == '__main__':
    pass
