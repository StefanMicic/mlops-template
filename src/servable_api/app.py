from configparser import ConfigParser

import keras
from flask import Flask, request

app = Flask(__name__)

cfg = ConfigParser()
cfg.read('../config.ini')
model = keras.models.load_model(cfg.get('train', 'model_path'))


@app.route('/predict', methods=['POST'])
def model_predict():
    input_data = request.form['input']
    predict = model(request.form['model_path']).predict(input_data)
    return predict


if __name__ == '__main__':
    app.run(debug=True)
