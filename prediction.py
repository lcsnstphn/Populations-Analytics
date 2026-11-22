import numpy as np
import pandas as pd

def make_single_prediction(model, scaler, age, population, is_youth):
    input_features = np.array([[age, population, is_youth]])
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    return prediction, probability
