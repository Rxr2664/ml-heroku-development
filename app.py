from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    result = "Setosa 🌸" if prediction == 1 else "Not Setosa 🌿"
    return render_template('index.html', prediction_text=f"Prediction: {result}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    values = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(values)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
