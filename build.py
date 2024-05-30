import pandas as pd
data = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\FourthSEM\concatenated_140.csv")
data['process'].unique()
X= data[['vdd','pd','vinp','temperature','process']]
X

print(data)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
X['process'] = label_encoder.fit_transform(X['process'])
print(X)

y = data['vinn']
y

x1 = X[(X['pd'] > 0) & (X['vinp'] == 0)]
x2 = X[(X['pd'] == 0) & (X['vinp'] == 0)]
x3 = X[(X['pd'] == 0) & (X['vinp'] > 0) ]
x4 = X[(X['pd'] > 0) & (X['vinp'] > 0) ]


y1 = y[(X['pd'] > 0) & (X['vinp'] == 0)]
y2 = y[(X['pd'] == 0) & (X['vinp'] == 0)]
y3 = y[(X['pd'] == 0) & (X['vinp'] > 0) ]
y4 = y[(X['pd'] > 0) & (X['vinp'] > 0) ]


from sklearn.ensemble import RandomForestRegressor

model2 = RandomForestRegressor(n_estimators=5,max_depth=11) 

# Train the model on the training data
model2.fit(x1, y1)
model3 = RandomForestRegressor(n_estimators=5,max_depth=11) 

# Train the model on the training data
model3.fit(x2, y2)
model4 = RandomForestRegressor(n_estimators=5,max_depth=11) 

# Train the model on the training data
model4.fit(x3, y3)
model5 = RandomForestRegressor(n_estimators=5,max_depth=11) 

# Train the model on the training data
model5.fit(x4, y4)
model_data2 = {
    'model': model2,
    'feature_names':  list(x1.columns),
    'label_encoder': label_encoder
}
x1.columns
model_data3 = {
    'model': model3,
    'feature_names':  list(x2.columns),
    'label_encoder': label_encoder
}
x2.columns
model_data4 = {
    'model': model4,
    'feature_names':  list(x3.columns),
    'label_encoder': label_encoder
}
x3.columns

model_data5 = {
    'model': model5,
    'feature_names':  list(x4.columns),
    'label_encoder': label_encoder
}
x4.columns


# Save the feature names along with the model and label encoder using pickle
import pickle

with open('./trained_model/Regions_1.pkl', 'wb') as file:
    pickle.dump(model_data2, file)



with open('./trained_model/Regions_2.pkl', 'wb') as file:
    pickle.dump(model_data3, file)




with open('./trained_model/Regions_3.pkl', 'wb') as file:
    pickle.dump(model_data4, file)



#

with open('./trained_model/Regions_4.pkl', 'wb') as file:
    pickle.dump(model_data5, file)

print("Program Executed")