import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model 
from tensorflow.keras.metrics import MeanSquaredError

def extract_and_save_weights(model_path):
    try:
        model = load_model(model_path, custom_objects={'MeanSquaredError': MeanSquaredError, 'mse': 'mean_squared_error'})
        w1 = model.layers[0].get_weights()[0]
        b1 = model.layers[0].get_weights()[1]
        w2 = model.layers[1].get_weights()[0]
        b2 = model.layers[1].get_weights()[1]
        w3 = model.layers[2].get_weights()[0]
        b3 = model.layers[2].get_weights()[1]
        weights = [w1, b1, w2, b2, w3, b3]
        folder_name = 'mlp_saved_weights'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        # Save weights and biases as CSV
        np.savetxt(os.path.join(folder_name, 'w1.csv'), w1, delimiter=',')
        np.savetxt(os.path.join(folder_name, 'b1.csv'), b1, delimiter=',')
        np.savetxt(os.path.join(folder_name, 'w2.csv'), w2, delimiter=',')
        np.savetxt(os.path.join(folder_name, 'b2.csv'), b2, delimiter=',')
        np.savetxt(os.path.join(folder_name, 'w3.csv'), w3, delimiter=',')
        np.savetxt(os.path.join(folder_name, 'b3.csv'), b3, delimiter=',')
        
        # Save weights and biases as matrices
        np.savetxt(os.path.join(folder_name, 'w1_matrix.txt'), w1)
        np.savetxt(os.path.join(folder_name, 'b1_matrix.txt'), b1)
        np.savetxt(os.path.join(folder_name, 'w2_matrix.txt'), w2)
        np.savetxt(os.path.join(folder_name, 'b2_matrix.txt'), b2)
        np.savetxt(os.path.join(folder_name, 'w3_matrix.txt'), w3)
        np.savetxt(os.path.join(folder_name, 'b3_matrix.txt'), b3)
        
        print("Weights and biases extracted and saved successfully!")
        return weights
    except (FileNotFoundError, IndexError):
        print("Error: Unable to load model or extract weights and biases.")
        return None

if __name__ == "__main__":
    model_path = input("Enter the path to the model file: ")
    weights = extract_and_save_weights(model_path)
    if not weights:
        print("Error: Unable to extract weights and biases.")
