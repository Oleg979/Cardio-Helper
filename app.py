import json

from utils.predict import predict
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/predict', methods=['POST'])
@cross_origin()
def classify():
    print("Initializing backend...")
    print("Loading network...")
    body = request.json
    print(body)
    probability = predict(body['data'])
    print(probability)
    return jsonify({'success': True, 'result': json.dumps(probability.astype(float))})


@app.route('/main', methods=['GET'])
@cross_origin()
def root():
    print("Serving main file...")
    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key = 'secret'
    app.run(debug=True)
