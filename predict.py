import pickle

from flask import Flask, request, jsonify

model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('airline-satisfaction')

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'ok', 'usage': 'POST passenger data to /predict'})


@app.route('/predict', methods=['POST'])
def predict():
    passenger = request.get_json()

    X = dv.transform([passenger])
    probability = float(model.predict_proba(X)[0, 1])

    return jsonify({
        'satisfaction_probability': probability,
        'satisfied': probability >= 0.5,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)