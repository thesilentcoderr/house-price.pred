import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('house_price.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    area = float(request.args.get('area'))
    prediction = model.predict([[area]])
    return render_template('index.html', prediction_text='Regression Model  has predicted Price for given Area is : {}'.format(prediction))