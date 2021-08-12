import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import os
from datetime import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# init app
app = Flask(__name__)

# DataBase
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:admin@127.0.0.1:3306/lcm"
db = SQLAlchemy(app)

ma = Marshmallow(app)

# data preprocess - do normalization
def data_preprocess(data):
    # Normalization
    data['NTSC'] = data['NTSC'] / 99
    data['brightness'] = data['brightness'] / 700
    data['gamma'] = (data['gamma'] - 1.8) / (2.6 - 1.8)
    data['temperature'] = (data['temperature'] - 1000) / (10000 - 1000)
    return data

# TestData Class/Model
class TestData(db.Model):
    __tablename__ = 'test_datas'
    id = db.Column(db.Integer, primary_key=True)
    lcm_id = db.Column(db.String(255), unique=True, nullable=False)
    model = db.Column(db.String(255), nullable=False)
    ntsc = db.Column(db.Float, nullable=False)
    gamma = db.Column(db.Float, nullable=False)
    brightness = db.Column(db.Float, nullable=False)
    temperature  =  db.Column(db.Float, nullable=False)
    level = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, onupdate=datetime.now, default=datetime.now)

    def __init__(self, lcm_id, model, ntsc, gamma, brightness, temperature, level):
        self.lcm_id = lcm_id
        self.model = model
        self.ntsc = ntsc
        self.gamma = gamma
        self.brightness = brightness
        self.temperature = temperature
        self.level = level

class TestDataSchema(ma.Schema):
    class Meta:
        fields = ('id', 'lcm_id', 'model', 'ntsc', 'gamma', 'brightness', 'temperature', 'level')

testData_Schema = TestDataSchema()
testDatas_Schema = TestDataSchema(many = True)

@app.route('/', methods=['GET'])
def get():
    return jsonify({ 'msg': 'Hello' })

@app.route('/post', methods=['POST'])
def post():
    lcm_id = request.json['id']
    lcm_model = request.json['model']
    ntsc = request.json['NTSC']
    gamma = request.json['gamma']
    brightness = request.json['brightness']
    temperature = request.json['temperature']

    data = data_preprocess({'NTSC': float(ntsc), 'brightness': float(brightness), 'gamma': float(gamma), 'temperature': float(temperature)})
    data = [data['NTSC'], data['brightness'], data['gamma'], data['temperature']]
    model = tf.keras.models.load_model('LCM_model.h5')
    predictions = model.predict(np.reshape(data, (1, 4)))
    result = np.argmax(predictions) + 1

    check_data_exeits = TestData.query.filter(TestData.lcm_id == lcm_id).scalar()
    
    if check_data_exeits is None:
        new_test_data = TestData(lcm_id, lcm_model, ntsc, gamma, brightness, temperature, result)
        db.session.add(new_test_data)
        db.session.commit()
        return jsonify({ 'status': 'ok', 'level': str(result)})
    else:
        check_data_exeits.ntsc = ntsc
        check_data_exeits.gamma = gamma
        check_data_exeits.brightness = brightness
        check_data_exeits.temperature = temperature
        check_data_exeits.level = result

        db.session.commit()
        return jsonify({ 'status': 'ok', 'level': str(result)})


@app.route('/getOpticalData/<id>', methods=['GET'])
def get_Optical_data(id):
    data = TestData.query.filter(TestData.lcm_id == id).scalar()

    # print(test)
    if data is None:
        return jsonify({ 'msg': 0 })
    else:
        return testData_Schema.jsonify(data)

# Run Server
if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True, host='chengkai.nfu.edu.tw', ssl_context=('certificate.crt', 'private.key'))