import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.binary_encoder = ce.BinaryEncoder(cols=['Vehicle Class', 'Transmission'])
        self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

    def fit(self, X, y=None):
        X_temp = self.binary_encoder.fit_transform(X)
        self.onehot_encoder.fit(X_temp[['Fuel Type']])
        return self

    def transform(self, X):
        X_temp = self.binary_encoder.transform(X)
        fuel_encoded = pd.DataFrame(
            self.onehot_encoder.transform(X_temp[['Fuel Type']]),
            columns=self.onehot_encoder.get_feature_names_out(['Fuel Type']),
            index=X_temp.index
        )
        X_temp = X_temp.drop(columns=['Fuel Type']).reset_index(drop=True)
        fuel_encoded = fuel_encoded.reset_index(drop=True)
        return pd.concat([X_temp, fuel_encoded], axis=1)
