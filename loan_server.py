# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 11:40:47 2020

@author: padma
"""

from flask import Flask, jsonify, request
from waitress import serve
import json
import requests

import os
import pandas as pd
from sklearn.externals import joblib
import dill as pickle

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello Padma!"

@app.route('/predict_old', methods=['POST'])
def apicall_orig():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

        #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        #Getting the Loan_IDs separated out
        loan_ids = test['Loan_ID']

    except Exception as e:
        raise e

    clf = 'loan_pred_v1.pk'

    if test.empty:
        return(bad_request())
    else:
        #Load the saved model
        print("Loading the model...")
        loaded_model = None
        with open('./trained_models/'+clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)
    
@app.route('/predict', methods=['POST'])   
def apicall():
    """API Call

    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient='records')

        #To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        #Getting the Loan_IDs separated out
        loan_ids = test['Loan_ID']

    except Exception as e:
        raise e

    clf = 'loan_pred_v1.pk'

    if test.empty:
        return(bad_request())
    else:
        #Load the saved model
        print("Loading the model...")
        loaded_model = None
        with open('./trained_models/'+clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        #prediction_series = list(pd.Series(predictions))

        #final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))
        pred_df = pd.DataFrame(list(zip(loan_ids, predictions)), columns=['Loan_ID', 'Loan_Status'])
        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        print(pred_df)
        responses = jsonify(predictions=pred_df.to_json(orient="records"))
        responses.status_code = 200

        return (responses)
    
@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

if __name__ == "__main__":
 #app.run()
    serve(app, host='0.0.0.0', port=5003)