import sys
from src.exception import CustomException
from src.logger import logging

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.component.cost_optimization import TariffLoadShiftOptimizer

if __name__ == '__main__':
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        mae, rmse, r2, mape = model_trainer.initate_model_training(train_arr, test_arr)

        print(f'Final Test MAE : {mae}')
        print(f'Final Test RMSE : {rmse}')
        print(f'Final Test R2 Score : {r2}')
        print(f'Final Test MAPE : {mape:.2f}%')

        # Tariff-based load-shifting cost simulation, run on the full
        # ingested dataset so it reflects the plant's actual load profile
        optimizer = TariffLoadShiftOptimizer()
        optimizer.run()

    except Exception as e:
        logging.info('Exception occured in training pipeline')
        raise CustomException(e, sys)
