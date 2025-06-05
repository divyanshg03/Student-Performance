import pickle
from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__, template_folder='template')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'GET':
        return render_template('home_enhanced.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethinicity=request.form.get('race_ethinicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        res= predict_pipeline.predict(pred_df)
        return render_template('home_enhanced.html', results=res[0])

if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug = True, port = 5000)    