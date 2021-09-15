import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

path_model_RF_1 = './models/model_RF_1.pkl'
RF_1 = joblib.load(path_model_RF_1)
path_model_RF_2 = './models/model_RF_2.pkl'
RF_2 = joblib.load(path_model_RF_2)
path_model_RF_3 = './models/model_RF_3.pkl'
RF_3 = joblib.load(path_model_RF_3)
path_model_RF_4 = './models/model_RF_4.pkl'
RF_4 = joblib.load(path_model_RF_4)
path_model_RF_5 = './models/model_RF_5.pkl'
RF_5 = joblib.load(path_model_RF_5)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    prediction_RF_1 = RF_1.predict_proba(final_features)
    prediction_RF_2 = RF_2.predict_proba(final_features)
    prediction_RF_3 = RF_3.predict_proba(final_features)
    prediction_RF_4 = RF_4.predict_proba(final_features)
    prediction_RF_5 = RF_5.predict_proba(final_features)

    prediction = prediction_RF_1 + prediction_RF_2 + prediction_RF_3 + prediction_RF_4 + prediction_RF_5
    prediction_mean = prediction / 5
    syncope_type = ["Cardiogenic", "Other Types"]
    if prediction_mean[:,0] > prediction_mean[:,1]:
        syncope = syncope_type[0]
    else:
        syncope = syncope_type[1]

    return render_template('index.html', prediction_text = "Syncope Type is {}" .format(syncope))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)