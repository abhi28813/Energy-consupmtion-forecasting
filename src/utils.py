import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(xtrain,ytrain,xtest,ytest,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(xtrain,ytrain)

            # Predict Training data
            y_train_pred = model.predict(xtrain)

            # Predict Testing data
            y_test_pred =model.predict(xtest)

            # Get R2 scores for train and test data
            train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(ytest,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)

def mape_score(true, predicted, epsilon=1.0):
    '''
    Mean Absolute Percentage Error. `epsilon` floors the denominator so
    near-zero actual readings (idle/no-load periods) don't blow up the
    average with divide-by-near-zero percentage errors.
    '''
    try:
        true = np.asarray(true, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        denom = np.where(np.abs(true) < epsilon, epsilon, np.abs(true))
        return float(np.mean(np.abs((true - predicted) / denom)) * 100)
    except Exception as e:
        logging.info('Exception Occured while evaluating MAPE')
        raise CustomException(e,sys)

def model_metrics(true, predicted):
    try :
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        mape = mape_score(true, predicted)
        return mae, rmse, r2_square, mape
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)

def seasonal_naive_baseline(train_df, test_df, target_col='Usage_kWh',
                             group_cols=('Load_Type', 'WeekStatus', 'hour_bin')):
    '''
    Naive 'no-model' baseline forecast: the historical average usage for
    the same Load_Type / WeekStatus / 2-hour time-of-day bucket, learned
    from the training data. Mirrors how the plant would estimate demand
    today without any ML model ("same as it usually is at this time").
    Used to benchmark the trained model's MAPE improvement.
    '''
    try:
        train = train_df.copy()
        test = test_df.copy()
        train['hour_bin'] = (train['hours'] // 2 * 2).astype(int)
        test['hour_bin'] = (test['hours'] // 2 * 2).astype(int)

        group_cols = list(group_cols)
        lookup = train.groupby(group_cols)[target_col].mean()
        overall_mean = train[target_col].mean()

        keys = list(test[group_cols].itertuples(index=False, name=None))
        return np.array([lookup.get(key, overall_mean) for key in keys])
    except Exception as e:
        logging.info('Exception Occured while computing seasonal naive baseline')
        raise CustomException(e,sys)

def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2, model_train_mape = model_metrics(ytrain, ytrain_pred)
        model_test_mae , model_test_rmse, model_test_r2, model_test_mape = model_metrics(ytest, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))
        print("- MAPE: {:.2f}%".format(model_train_mape))

        print('----------------------------------')

        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        print("- MAPE: {:.2f}%".format(model_test_mape))

    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)