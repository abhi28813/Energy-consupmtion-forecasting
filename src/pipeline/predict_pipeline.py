import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e,sys)


class CustomData:
    def __init__(self,
                 Leading_Current_Reactive_Power_kVarh:float,
                 Lagging_Current_Power_Factor:float,
                 Leading_Current_Power_Factor:float,
                 NSM:int,
                 hours:float,
                 WeekStatus:str,
                 Day_of_week:str,
                 Load_Type:str,
                 Month:str):

        self.Leading_Current_Reactive_Power_kVarh = Leading_Current_Reactive_Power_kVarh
        self.Lagging_Current_Power_Factor = Lagging_Current_Power_Factor
        self.Leading_Current_Power_Factor = Leading_Current_Power_Factor
        self.NSM = NSM
        self.hours = hours
        self.WeekStatus = WeekStatus
        self.Day_of_week = Day_of_week
        self.Load_Type = Load_Type
        self.Month = Month

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Leading_Current_Reactive_Power_kVarh':[self.Leading_Current_Reactive_Power_kVarh],
                'Lagging_Current_Power_Factor':[self.Lagging_Current_Power_Factor],
                'Leading_Current_Power_Factor':[self.Leading_Current_Power_Factor],
                'NSM':[self.NSM],
                'hours':[self.hours],
                'WeekStatus':[self.WeekStatus],
                'Day_of_week':[self.Day_of_week],
                'Load_Type':[self.Load_Type],
                'Month':[self.Month]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
