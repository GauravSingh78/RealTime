import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

# Load data
# file = r"C:\Users\ASUS\OneDrive\Desktop\FourthSEM\concatenated_140.csv"
# df = pd.read_csv(file)
# print(df)


# df_train= pd.get_dummies(df, columns = ['process'])
# df = df_train.replace({True: 1, False: 0})
# df_train.to_csv("one_forty_train_onehotencoded.csv")

file = "one_forty_train_onehotencoded.csv"
df = pd.read_csv(file)
print(df)

# Define input features
inputs = ['vinp', 'pd', 'vdd', 'temperature', 'process_fastnfastp',
          'process_fastnslowp', 'process_slownfastp', 'process_slownslowp',
          'process_typical', 'vinn']

# Preprocess data
df_train1 = df[inputs]

X = df_train1.drop('vinn', axis=1).values
y = df_train1['vinn'].values



from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


model = Sequential()

model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='Adam',loss=MeanSquaredError(), metrics=['mae', 'mse', 'accuracy'])

model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=64,epochs=10)
model.summary()

y_pred = model.predict(X_test)
model.save("trained_model/mlp.h5")
print ("model saved")


#print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
#print('MSE:', metrics.mean_squared_error(y_test, y_pred))
#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print('VarScore:',metrics.explained_variance_score(y_test,y_pred))







# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
# from sklearn import metrics
# import warnings
# warnings.filterwarnings("ignore")


# file = "one_forty_train_onehotencoded.csv"
# df = pd.read_csv(file)
# print (df.head())
# inputs = ['vinp','xpd','vdd','temperature','process_fastnfastp','process_fastnslowp','process_slownfastp','process_slownslowp','process_typical','vinn']

# df_train = df[inputs]
# print(df_train.columns)
# print("total rows ",df_train.count)

# df_train1 = pd.DataFrame()
# df_train1 = df_train[inputs]
# print (df_train1.columns)
# print (df_train1.head())


# X = df_train1.drop('vinn',axis =1).values
# y = df_train1['vinn'].values

# print(X.shape)
# print(y.shape)


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# model = Sequential()
# #model.add(Dense(32,activation='relu'))
# #model.add(Dense(8,activation='relu'))
# #model.add(Dense(1))
# #model.add(Dense(16,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1))

# model.compile(optimizer='Adam',loss='mse', metrics=['mae','mse','accuracy'])

# model.fit(x=X_train,y=y_train,
#           validation_data=(X_test,y_test),
#           batch_size=128,epochs=10)
# model.summary()

# y_pred = model.predict(X_test)
# model.save("./trained_model/mlp.h5")
# print ("model saved")


# print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# print('MSE:', metrics.mean_squared_error(y_test, y_pred))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# print('VarScore:',metrics.explained_variance_score(y_test,y_pred))