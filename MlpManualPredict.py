import pandas as pd
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Desktop\fastnfastp\test\one_hot_fastnfastp_3.6V_45.csv")
df

inputs =['vinp', 'pd', 'vdd','temperature','process_fastnfastp','process_fastnslowp', 'process_slownfastp', 'process_slownslowp','process_typical','vinn']
data =df[inputs]
X_test = data.drop('vinn', axis=1).values


import numpy as np
import h5py

import tensorflow as tf

model = tf.keras.models.load_model("neuron_model.h5")

def extract_weights_and_biases(model):
    
    try:
        w1 = model.layers[0].get_weights()[0]
        b1 = model.layers[0].get_weights()[1]
        w2 = model.layers[1].get_weights()[0]
        b2 = model.layers[1].get_weights()[1]
        w3 = model.layers[2].get_weights()[0]
        b3 = model.layers[2].get_weights()[1]
        return w1, b1, w2, b2, w3, b3
    except IndexError:
        print("Warning: Model architecture might not match expected structure.")
        return None


weights = extract_weights_and_biases(model)

if weights:
    w1, b1, w2, b2, w3, b3 = weights
    print("Weights and biases extracted successfully!")
else:
    print("Error: Unable to extract weights and biases.")




def forward_propagation(inputs, weights):
    
    w1, b1, w2, b2, w3, b3 = weights
    print("Shape of b1:", b1.shape)
    print("Shape of w1:", w1.shape)
    print("Shape of b2:", b2.shape)
    print("Shape of w2:", w2.shape)
    print("Shape of b3:", b3.shape)
    print("Shape of w3:", w3.shape)
    
    b1_reshaped = b1.reshape(1, -1)
    b2_reshaped = b2.reshape(1, -1)
    b3_reshaped = b3.reshape(1, -1)
    # Perform forward propagation
    print("Shape of b1_reshaped:", b1_reshaped.shape)
    print("Shape of b2_reshaped:", b2_reshaped.shape)
    print("Shape of b3_reshaped:", b3_reshaped.shape)
    print("Shape of input",inputs.shape)
    z1 = np.dot(inputs, w1) + b1_reshaped
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2_reshaped
    a2 = relu(z2)
    z3 = np.dot(a2, w3) + b3_reshaped
    print("Shape of z1:", z1.shape)
    print("Shape of z2:", z2.shape)
    print("Shape of z3:", z3.shape)

    
    return z3

def relu(x):
    return np.maximum(0, x)





# arr = np.array([[0.0, 0.0, 0.0066, -15, 1, 0, 0, 0, 0]])
# arr = np.array([[0.000000,	0.0066,	0.0066,	-15,	1,	0,	0,	0,	0	]])
# arr = np.array([[1.647645,	3.0000,	3.0000,	85,	0,	0,	0,	0,	1]])
predictions = forward_propagation(X_test, weights)

print("Predictions:", predictions)


print(len(predictions))

import matplotlib.pyplot as plt

vinn = df['vinn']

plt.plot(vinn,label='Actual',color='blue')
plt.plot(predictions,label="Predicted", color='red')
plt.title("Manual Prediction")
plt.legend()
plt.tight_layout()
plt.show()