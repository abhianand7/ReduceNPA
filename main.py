from data_handling.read_and_clean_data import check_data, remove_features, clean_data
from data_handling.prepare_and_analyze import get_training_data
from models.linear_model import run_ktimes
from models.boost_model import run_xgb
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb


def train(data_file_path):
    """

    :param data_file_path:
    :return:
    """
    df = check_data(data_file_path)
    df, removed_features = remove_features(df)
    df = clean_data(df)
    x, y, median_dict, category_freq_value_dict, mean_enc_dicts, to_scaled_features, categorical_features, training_cols = get_training_data(df)

    # test linear classifier model
    # run_ktimes(x, y)

    # test xgboost model
    xgb_model = run_xgb(x, y)

    return xgb_model, removed_features, median_dict, category_freq_value_dict, mean_enc_dicts, to_scaled_features, categorical_features, training_cols


def test_model(test_file, model, scaler, median_dict, category_freq_value_dict, mean_enc_dicts, removed_features, to_scaled_features, categorical_features, training_cols):
    """
    test trained model on test dataset
    :param model:
    :param scaler:
    :param median_dict:
    :param category_freq_value_dict:
    :param mean_enc_dicts:
    :param removed_features:
    :return:
    """
    # prepare data for testing
    # use the same scaler used for transforming train dataset
    with open(scaler, 'rb') as fobj:
        scaler_obj = pickle.load(fobj)

    # read raw test file
    df = check_data(test_file)

    # drop features which were dropped in training for having too many missing values
    df.drop(labels=removed_features, axis=1, inplace=True)
    df = clean_data(df)

    # drop other non-essential features
    cols_to_drop = ['title', 'emp_title', 'batch_enrolled']
    df.drop(labels=cols_to_drop, axis=1, inplace=True)
    # handle missing values

    # fill the missing values in the test dataset using the same transformation as of training
    for i in to_scaled_features:
        if i in cols_to_drop:
            continue
        df[i].fillna(value=median_dict[i], inplace=True)

    for i in categorical_features:
        if i in cols_to_drop:
            continue
        df[i].fillna(value=category_freq_value_dict[i], inplace=True)

    # scale features
    scaled_data = scaler_obj.transform(df[to_scaled_features])
    scaled_df = pd.DataFrame(scaled_data, columns=to_scaled_features)
    columns = df.columns
    other_columns = list((set(columns) - set(to_scaled_features)))
    non_scaled_df = df[other_columns]
    new_df = scaled_df.join(non_scaled_df)

    # delete unused data frames
    del scaled_df, non_scaled_df, df

    # handling encoding of categorical features, and apply the same transformation as of training
    for i in categorical_features:
        new_df[i] = new_df[i].map(mean_enc_dicts[i])

    # inputs
    input_features = categorical_features + to_scaled_features

    member_ids = new_df['member_id'].values
    new_df.drop(labels=['member_id'], axis=1, inplace=True)
    new_df.info()
    input_features = new_df[input_features]
    input_features = input_features.reindex(columns=training_cols)
    x = np.array(input_features)
    # y = np.array(new_df[['loan_status']])

    xgb_dmatrix = xgb.DMatrix(x)
    predictions = model.predict(xgb_dmatrix)
    predictions = [1 if i >= 0.5 else 0 for i in predictions]

    # save the predicted results
    with open('submission.csv', 'w') as fobj:
        fobj.write('member_id,loan_status\n')
        for i, j in zip(predictions, member_ids):
            fobj.write("{},{}\n".format(j, i))


def run(train_file, test_file):
    """
    provide training and testing file and run the model
    :param train_file:
    :param test_file:
    :return:
    """

    model, removed_features, median_dict, category_freq_value_dict, mean_enc_dicts, to_scaled_features, categorical_features, training_cols = train(train_file)

    test_model(test_file, model, 'scaler.pkl', median_dict, category_freq_value_dict, mean_enc_dicts, removed_features,
               to_scaled_features, categorical_features, training_cols)


if __name__ == '__main__':
    # provide the path to training file
    dataset_file_path = 'dataset/train_indessa.csv'
    # provide the path to testing file
    test_file_path = 'dataset/test_indessa.csv'
    run(dataset_file_path, test_file_path)
