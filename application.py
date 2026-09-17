from datetime import datetime

from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

LOAD_TYPES = ['Light_Load', 'Medium_Load', 'Maximum_Load']

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html', load_types=LOAD_TYPES)
    else:
        form_data = request.form
        try:
            reading_datetime = datetime.fromisoformat(form_data.get('reading_datetime'))
            nsm = reading_datetime.hour * 3600 + reading_datetime.minute * 60 + reading_datetime.second
            is_weekend = reading_datetime.weekday() >= 5

            data = CustomData(
                Leading_Current_Reactive_Power_kVarh=float(form_data.get('Leading_Current_Reactive_Power_kVarh')),
                Lagging_Current_Power_Factor=float(form_data.get('Lagging_Current_Power_Factor')),
                Leading_Current_Power_Factor=float(form_data.get('Leading_Current_Power_Factor')),
                NSM=nsm,
                hours=nsm / 3600,
                WeekStatus='Weekend' if is_weekend else 'Weekday',
                Day_of_week=reading_datetime.strftime('%A'),
                Load_Type=form_data.get('Load_Type'),
                Month=reading_datetime.strftime('%B')
            )
            pred_df=data.get_data_as_dataframe()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results=predict_pipeline.predict(pred_df)
            print("after Prediction")
            return render_template('home.html', load_types=LOAD_TYPES, form_data=form_data, prediction=round(results[0], 2))
        except Exception as e:
            return render_template('home.html', load_types=LOAD_TYPES, form_data=form_data, error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)        


