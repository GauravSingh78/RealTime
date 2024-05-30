import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\FourthSEM\concatenated_140.csv")

print(data.shape)
X= data[['vdd','pd','vinp','temperature','process']]
X
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X['process'] = label_encoder.fit_transform(X['process'])


y = data['vinn']
y
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


rf_regressor = RandomForestRegressor(n_estimators=10,  max_depth=11)

# Train the model on the training data
rf_regressor.fit(X, y)



import pickle

load_model= {
    'model':rf_regressor,
    'label_encoder':label_encoder,
    'feature_names':list(X.columns)
}

# Save the trained model to a pickle file
with open('Random_Forest.pkl', 'wb') as file:
    pickle.dump(load_model, file)
    print("Model dumped into pickle file")