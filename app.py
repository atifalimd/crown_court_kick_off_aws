import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from flask import Flask, request,render_template,jsonify
import cloudpickle as pickle

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print('Data is: ')
    output=model.predict(pd.DataFrame([data]))
    print('output is',output[0])
    return jsonify(int(output[0]))

@app.route('/predict',methods=['POST'])
def predict():
    Gender=request.form.get('Gender')
    Age=request.form.get('Age')
    offence=request.form.get('offence')
    Sentence=request.form.get('Sentence')
    court_room=request.form.get('court_room')
    temp_max=int(request.form.get('temp_max'))
    temp_min =int(request.form.get('temp_min'))
    month =request.form.get('month')
    reason=request.form.get('reason')
    output=model.predict(pd.DataFrame([[Gender,Age,offence,Sentence,court_room,temp_max,temp_min,month,reason]],
                        columns=['Gender', 'Age', 'offence', 'Sentence', 'court_room', 'temp_max',
                                          'temp_min', 'month', 'reason']))[0]
    return render_template('home.html',predict_text="The Prediction is {}".format(output))


if __name__=='__main__':
    app.run(debug=True)