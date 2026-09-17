
# Basic Import
import numpy as np
import pandas as pd

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
from src.utils import print_evaluated_results
from src.utils import model_metrics
from src.utils import mape_score
from src.utils import seasonal_naive_baseline

from dataclasses import dataclass
import json
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    metrics_report_file_path = os.path.join('artifacts','metrics_report.json')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            xtrain, ytrain, xtest, ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "GradientBoosting Regressor":GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(xtrain,ytrain,xtest,ytest,models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            # To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has r2 Score less than 60%')
                raise CustomException('No Best Model Found')

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            logging.info('Hyperparameter tuning started for catboost')

            # Hyperparameter tuning on Catboost
            # Initializing catboost
            cbr = CatBoostRegressor(verbose=False)

            # Creating the hyperparameter grid
            param_dist = {'depth'          : [4,5,6,7,8,9, 10],
                          'learning_rate' : [0.01,0.02,0.03,0.04],
                          'iterations'    : [300,400,500,600]}

            #Instantiate RandomSearchCV object
            rscv = RandomizedSearchCV(cbr , param_dist, scoring='r2', cv =5, n_jobs=-1)

            # Fit the model
            rscv.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best Catboost parameters : {rscv.best_params_}')
            print(f'Best Catboost Score : {rscv.best_score_}')
            print('\n====================================================================================\n')

            best_cbr = rscv.best_estimator_

            logging.info('Hyperparameter tuning complete for Catboost')

            logging.info('Hyperparameter tuning started for KNN')

            # Initialize knn
            knn = KNeighborsRegressor()

            # parameters
            k_range = list(range(2, 31))
            param_grid = dict(n_neighbors=k_range)

            # Fitting the cvmodel
            grid = GridSearchCV(knn, param_grid, cv=5, scoring='r2',n_jobs=-1)
            grid.fit(xtrain, ytrain)

            # Print the tuned parameters and score
            print(f'Best KNN Parameters : {grid.best_params_}')
            print(f'Best KNN Score : {grid.best_score_}')
            print('\n====================================================================================\n')

            best_knn = grid.best_estimator_

            logging.info('Hyperparameter tuning Complete for KNN')

            logging.info('Voting Regressor model training started')

            # Creating final Voting regressor
            er = VotingRegressor([('cbr',best_cbr),('xgb',XGBRegressor()),('knn',best_knn)], weights=[3,2,1])
            er.fit(xtrain, ytrain)
            print('Final Model Evaluation :\n')
            print_evaluated_results(xtrain,ytrain,xtest,ytest,er)
            logging.info('Voting Regressor Training Completed')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = er
            )
            logging.info('Model pickle file saved')
            # Evaluating Ensemble Regressor (Voting Regressor on test data)
            ytest_pred = er.predict(xtest)

            mae, rmse, r2, mape = model_metrics(ytest, ytest_pred)
            logging.info(f'Test MAE : {mae}')
            logging.info(f'Test RMSE : {rmse}')
            logging.info(f'Test R2 Score : {r2}')
            logging.info(f'Test MAPE : {mape}')

            # Benchmark against a naive 'no-model' baseline forecast so the
            # model's MAPE improvement is measurable, not asserted
            baseline_mape, improvement_pct = self._evaluate_baseline(ytest, mape)

            metrics_report = {
                'best_candidate_model': best_model_name,
                'best_candidate_r2': best_model_score,
                'final_model': 'VotingRegressor(CatBoost + XGBoost + KNN)',
                'test_mae': mae,
                'test_rmse': rmse,
                'test_r2': r2,
                'test_mape_pct': mape,
                'baseline_mape_pct': baseline_mape,
                'mape_improvement_pct': improvement_pct,
            }
            os.makedirs(os.path.dirname(self.model_trainer_config.metrics_report_file_path), exist_ok=True)
            with open(self.model_trainer_config.metrics_report_file_path, 'w') as f:
                json.dump(metrics_report, f, indent=2)
            logging.info(f'Metrics report saved to {self.model_trainer_config.metrics_report_file_path}')

            print(f'Baseline MAPE (no model) : {baseline_mape:.2f}%')
            print(f'Model MAPE                : {mape:.2f}%')
            print(f'MAPE improvement vs baseline : {improvement_pct:.2f}%')
            print('\n====================================================================================\n')

            logging.info('Final Model Training Completed')

            return mae, rmse, r2, mape

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

    def _evaluate_baseline(self, ytest, model_mape):
        '''
        Computes a seasonal-naive baseline forecast (historical average
        usage for the same Load_Type / WeekStatus / time-of-day bucket)
        from the raw train/test CSVs, and returns (baseline_mape,
        improvement_pct of the trained model over that baseline).
        '''
        try:
            train_df = pd.read_csv(self.model_trainer_config.train_data_path)
            test_df = pd.read_csv(self.model_trainer_config.test_data_path)

            baseline_pred = seasonal_naive_baseline(train_df, test_df)
            baseline_mape = mape_score(ytest, baseline_pred)
            improvement_pct = (baseline_mape - model_mape) / baseline_mape * 100

            logging.info(f'Baseline (seasonal-naive) MAPE : {baseline_mape}')
            logging.info(f'MAPE improvement over baseline : {improvement_pct}%')

            return baseline_mape, improvement_pct
        except Exception as e:
            logging.info('Exception occured while evaluating baseline forecast')
            raise CustomException(e,sys)
