import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics

from utils import Utils

# no mostrar warnings de sklearn
import warnings
warnings.filterwarnings("ignore")

class Models:

    def __init__(self):
        
        # Dividir los datos en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

        # Construcción de pipeline de escalado estándar y modelo para varios regresores.
        pipeline_lr=Pipeline([("scalar1",StandardScaler()),("lr_classifier",LinearRegression())])
        pipeline_dt=Pipeline([("scalar2",StandardScaler()),("dt_classifier",DecisionTreeRegressor())])
        pipeline_rf=Pipeline([("scalar3",StandardScaler()),("rf_classifier",RandomForestRegressor())])
        pipeline_kn=Pipeline([("scalar4",StandardScaler()),("rf_classifier",KNeighborsRegressor())])
        pipeline_xgb=Pipeline([("scalar5",StandardScaler()),("rf_classifier",XGBRegressor())])

        # lista de los pipelines
        pipelines = [pipeline_lr, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

        # Diccionario de pipelines y tipos de modelos
        pipe_dict = {0: "LinearRegression", 1: "DecisionTree", 2: "RandomForest",3: "KNeighbors", 4: "XGBoost"}

        # ajusto los pipelines
        for pipe in pipelines:
             pipe.fit(X_train, y_train)

        # Valores de RMSE para los diferentes modelos
        cv_results_rms = []
        for i, model in enumerate(pipelines):
            cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=10)
            cv_results_rms.append(cv_score)
            print("%s: %f " % (pipe_dict[i], cv_score.mean()))

        # Elijo el mejor modelo basado en el que tenga menor neg_root_mean_squared_error
        best_model = pipelines[np.argmax(cv_results_rms)]
        best_score = cv_results_rms[np.argmax(cv_results_rms)].mean()

        # Predicción del conjunto de prueba
        pred = best_model.predict(X_test)

        # Evaluacion del modelo
        print("R^2:",metrics.r2_score(y_test, pred))
        print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
        print("MAE:",metrics.mean_absolute_error(y_test, pred))
        print("MSE:",metrics.mean_squared_error(y_test, pred))
        print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, pred)))
        
        # Exporto el mejor modelo
        utils = Utils()
        utils.model_export(best_model, best_score)
        print(f'el mejor modelo es {pipe_dict[np.argmax(cv_results_rms)]} con un score de {best_score}')