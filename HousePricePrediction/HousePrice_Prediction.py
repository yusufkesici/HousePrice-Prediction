################################################
# End-to-End HousePrice Machine Learning Pipeline III
################################################

import joblib
import pandas as pd

df = pd.read_csv("HousePricePrediction/train.csv")

from HousePrice_pipeline import HousePrice_dataPred

X, y = HousePrice_dataPred(df)

random_user = X.sample(5, random_state=50)

new_model = joblib.load("voting_clf4.pkl")

new_model.predict(random_user)
