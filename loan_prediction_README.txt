Loan Prediction 
uses the following 3 files:
•	loan_pred_using_pipeline_flask.py :
     preprocesses data + fits the model + saves the model to a pickle file at    
     ./trained_models/loan_pred_v1.pk 
•	loan_server.py: 
     runs the Flask server waiting for a POST request to the 'predict' endpoint;
     converts the json request to a dataframe; 
     loads the estimator from the *.pkl file
     calls the grid.predict() on the test_df to generate the predictions.
    converts the predictions back to a dataframe.
•	loan_flask_predict.py: 
    Reads in the loan_prediction_test.csv; 
    converts it to json; 
    calls the predict endpoint to get the predictions.

Prediction :
preds = resp.json()['predictions']
Out[161]: '[{"Loan_ID":"LP001015","Loan_Status":1},{"Loan_ID":"LP001022","Loan_Status":1},{"Loan_ID":"LP001031","Loan_Status":1},{"Loan_ID":"LP001035","Loan_Status":1},{"Loan_ID":"LP001051","Loan_Status":1}]'

Note:
1) Ran Flask using waitress instead of gunicorn on Windows.

The following line changed in loan_server.py to use Waitress:
from waitress import serve
if __name__ == "__main__":
    #app.run()   # commented out
    serve(app, host='0.0.0.0', port=5003)  #used in place of the above

2)To run the Flask server, open a cmd terminal; activate py36_env; run :
python loan_server.py

3)test in a browser with : localhost:5003
it should print 'Hello Padma'

4) the requests.get/post call needed the url to be :
localhost:5003 instead of http://0.0.0.0:5003
which gave a ConnectionError + needed verify=False in the requests.post() call

5) Modified the code to use the dataframe instaed of converting the dataframes to matrices to pass as parameters to the estimator/pipeline

6) Used dill instead of joblib or pickle to save the modelmator
