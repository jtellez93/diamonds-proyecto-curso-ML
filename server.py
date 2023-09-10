import joblib
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

# creo la ruta para predecir
@app.route('/predict', methods=['GET'])

def predict():
    X_test = np.array([7.594444821,7.479555538,1.616463184,1.53352356,0.796666503,0.635422587,0.362012237,0.315963835,2.277026653])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion' : list(prediction)})


if __name__ == "__main__":

    # cargo el modelo
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)

# Para probar el modelo, ejecutamos el servidor y hacemos una petición GET a la ruta /predict: http://127.0.0.1:8080/predict