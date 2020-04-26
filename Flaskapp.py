import numpy as np
import pandas as pd
import sklearn
import pickle
import os
import sys
import requests
import json
import io
from flask import Flask,jsonify,request, render_template,redirect, send_file,Response

# Initialize the app and set a secret_key.
app = Flask(__name__)
app.secret_key = 'something_secret'

# Load the pickled model.
MODEL = pickle.load(open('model.pkl', 'rb'))

@app.route("/",methods=['GET', 'POST'])
def index():
    """Describe the model API inputs and outputs for users."""
    if request.method == 'POST':
        return redirect('127.0.0.1:8791/predict')
    else:
        return render_template('docs.html')


@app.route('/predict', methods=['POST'])
def user_show():
    uuid = request.form['user_id']
    df = pd.read_csv('data/predictd.csv')
    xnew = df[df['UniqueID']==int(uuid)]['predicted_loan_default']
    xnew.reset_index(inplace=True,drop=True)
    if(xnew[0]==0):
          return render_template('ld_0.html')
    else:
            return render_template('ld_1.html')
    return xnew.to_json(orient='split')




if __name__ == '__main__':
    app.run(debug=True,port=8791)

   


