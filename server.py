import joblib
import numpy as np
import pandas as pd

from utils import Utils
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify

utils_instance = Utils()

app = Flask(__name__)

# cargo el dataset
data = [
        {
        'carat': 0.23,
        'cut': 'Ideal',
        'color': 'E',
        'clarity': 'SI2',
        'depth': 61.5,
        'table': 55,
        'x': 3.95,
        'y': 3.98,
        'z': 2.43
        }
        ]
    
# convierto data en un dataframe
data = pd.DataFrame(data)
    
# Calculo volumen de los diamantes
data['volume'] = utils_instance.volume(data)

# Calculo la densidad de los diamantes
data['density'] = utils_instance.density(data)

# Aplicar codificador de etiquetas a cada columna con datos categóricos de acuerdo a las categorías del dataset
categorical_cols = ['cut', 'color', 'clarity']
cat_cut = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
cat_color = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
cat_clarity = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
label_encoder = LabelEncoder()
for col in categorical_cols:
    if col == 'cut':
        label_encoder.fit(cat_cut)
        data[col] = label_encoder.transform(data[col])
    elif col == 'color':
        label_encoder.fit(cat_color)
        data[col] = label_encoder.transform(data[col])
    else:
        label_encoder.fit(cat_clarity)
        data[col] = label_encoder.transform(data[col])

# creo la ruta para predecir
@app.route('/predict', methods=['GET'])

def predict():

    X_test = np.array(data)
    prediction = model.predict(X_test.reshape(1,-1))

    # Convertir el resultado de la predicción a float estándar
    prediction_float = float(prediction)
    
    return jsonify({'prediccion' : prediction_float})


if __name__ == "__main__":

    # cargo el modelo
    model = joblib.load('./models/best_model.pkl')
    app.run(port=8080)

# Para probar el modelo, ejecutamos el servidor y hacemos una petición GET a la ruta /predict: http://127.0.0.1:8080/predict