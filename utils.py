import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder

# creo la clase en nuestro entorno de trabajo
class Utils:

    def load_form_excel(self, path):
        return pd.read_excel(path)

    def volume(self, dataset):
        return dataset['x'] * dataset['y'] * dataset['z']

    def density(self, dataset):
        return dataset['carat'] / dataset['volume']        
    
    def clean_outliers(self, dataset, col):
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        return dataset[(dataset[col] >= Q1 - 1.5*IQR) & (dataset[col] <= Q3 + 1.5*IQR)]
    
    def label_encoder(self, dataset, cols):
        label_encoder = LabelEncoder()
        for col in cols:
            dataset[col] = label_encoder.fit_transform(dataset[col])
        return dataset
    
    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X, y
    
    def model_export(self, clf, score):
        joblib.dump(clf, './models/best_model.pkl')
        print(f'Modelo exportado con score {score}')