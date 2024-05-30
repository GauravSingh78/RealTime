import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model 
from tensorflow.keras.metrics import MeanSquaredError
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the test data
# df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\fastnfastp\test\one_hot_fastnfastp_3.6V_45.csv")
df = pd.read_csv('./mlpTest/fastnfastp_3.6V_45.csv')
# Select the relevant columns
# inputs = ['vinp', 'pd', 'vdd', 'temperature', 'process_fastnfastp', 'process_fastnslowp', 'process_slownfastp', 'process_slownslowp', 'process_typical', 'vinn']
# data = df[inputs]

# Prepare the test data
# X_test = data.drop('vinn', axis=1).values

# Load the trained model
# model_loaded = load_model('./trained_model/mlp_models.h5')
# model_loaded = load_model('./trained_model/neurons_model.h5', custom_objects={'MeanSquaredError': MeanSquaredError, 'mse': 'mean_squared_error'})

# # Compile the model with MeanSquaredError metric
# # model_loaded.compile(optimizer='adam', loss='mse', metrics=[MeanSquaredError()])

# # Predict the output
# predictions = model_loaded.predict(X_test)
# print(predictions)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("./trained_model/mlp.h5")

inputs = ['vinp', 'xpd', 'vdd', 'temperature', 'process_fastnfastp', 'process_fastnslowp', 
          'process_slownfastp', 'process_slownslowp', 'process_typical', 'vinn']
df_test1 = df[inputs]

# Prepare the input features for prediction
X_test = df_test1.drop('vinn', axis=1).values

# Make predictions
y_pred = model.predict(X_test)

# Print predictions
print("Predictions:")
print(y_pred)


# import matplotlib.pyplot as plt
# import pandas as pd
# from itertools import count
# from matplotlib.animation import FuncAnimation



# x_data = []
# y_data = []
# vinn = []

# index = count()

# def check_red_flag(y, vin):
#     if abs(y - vin) > 0.002:
#         return True
#     else:
#         return False

# def animate(i):
#     if next(index) < len(data):
#         x = df['time'].iloc[next(index)]
#         y = df['vinn'].iloc[next(index)]
#         vin = predictions[next(index)]
        
#         x_data.append(x)
#         y_data.append(y)
#         vinn.append(vin)
        
#         plt.cla()
#         plt.plot( y_data, label='Actual', color='b')
#         plt.plot( vinn, label='Predicted', color='red')
#         plt.xlabel('Time')
#         plt.ylabel('Voltage')
#         plt.title('Real-time Waveform Plot')
        
#         if check_red_flag(y, vin):
#             plt.text(x_data[-1], y_data[-1], 'Red Flag', color='red')
        
#         plt.legend()
#         plt.tight_layout()

# ani = FuncAnimation(plt.gcf(), animate, interval=1, cache_frame_data=False)

# plt.show()



# print(len(predictions))

import matplotlib.pyplot as plt

vinn = df['vinn']

plt.plot(vinn,label='Actual',color='blue')
plt.plot(y_pred,label="Predicted", color='red')
plt.title("Dot Predict Prediction")
plt.legend()
plt.tight_layout()
plt.show()