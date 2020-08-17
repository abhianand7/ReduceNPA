## Reduce NPA of a Bank

#### Approach
1. Model Selection: xgboost
2. Data Pre-processing Steps used
    1. Remove features with missing values larger than the threshold value(0.7) for the entire dataset
    2. Clean dataset
    3. Handle the missing values for numerical features
        1. Fill missing values with median of the features.
        2. Not using mean, in order to reduce the effect of outliers.
    4. Handle the missing values for Categorical features
        1. Used the most frequent value of each categorical feature to fill the missing values.
    5. Encoding categorical features using Target Encoding(Mean Encoding) for encoding the categories
        1. useful when the cardinality of categorical variable is very high.
        2. Though susceptible to over-fitting.
        3. use of regularization to prevent/reduce over-fitting.
    6. Scaling Features
        1. Considering each feature lies in a separate range, bring them to the same scale.
        2. used 'Standard Scaling' i.e. z score for scaling.
        3. useful in reducing the effects of outliers.
        
##### Steps to run the project
```
git clone https://github.com/abhianand7/ReduceNPA.git .
cd ReduceNPA
pip install -r requirements.txt
python main.py
```

##### Xgboost Model
###### parameters used
```
{'max_depth': 8, 'eta': 0.1, 'objective': 'binary:logistic', 'gamma': '0.3'}
boost_rounds = 200
```
###### train and test split used during training: 75:25
###### auc socre obtained: 0.9416 (Evaluation AUC score on test set)

##### Further Improvements:
1. Prepare a proper data pipeline so that the data pre-processings can be applied without any hassle.
2. Improvements on categorical features:
    1. handling less frequent categorical variables properly
    2. applying more feature engineering on categorical features
        1. possible option of using embeddings on text based features for adding more info to the model
    3. gridsearch for finding more optimal parameters for xgboost model
3. using better regularization techniques to prevent over-fitting
4. overall feature engineering requires more attention