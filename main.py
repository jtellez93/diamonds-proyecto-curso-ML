from utils import Utils
from models import Models

if __name__ == "__main__":

    # creo una instancia de la clase Utils
    utils_instance  = Utils()
    Models_instance = Models()

    # cargo el dataset
    data = utils_instance.load_form_excel('./data/diamonds.xlsx')

    # Retiro valores 0 en variables x, y, z
    data = data[(data['x'] != 0) & (data['y'] != 0) & (data['z'] != 0)]

    # Calculo volumen de los diamantes
    data['volume'] = utils_instance.volume(data)

    # Calculo la densidad de los diamantes
    data['density'] = utils_instance.density(data)

    # limpio outliers
    data_clean = utils_instance.clean_outliers(data, 'density')

    # Aplicar codificador de etiquetas a cada columna con datos categóricos
    label_data = utils_instance.label_encoder(data_clean, ['cut', 'color', 'clarity'])

    # separo features y target
    X, y = utils_instance.features_target(label_data, ['price'], ['price'])

    # entreno el modelo
    Models_instance.grid_training(X, y)