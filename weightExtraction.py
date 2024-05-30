import os
import numpy as np
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
    
    folder_name = 'saved_weights'
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
else:
    print("Error: Unable to extract weights and biases.")











# import os
# import numpy as np
# import tensorflow as tf

# model = tf.keras.models.load_model("neuron_model.h5")

# def extract_weights_and_biases(model):
#     try:
#         w1 = model.layers[0].get_weights()[0]
#         b1 = model.layers[0].get_weights()[1]
#         w2 = model.layers[1].get_weights()[0]
#         b2 = model.layers[1].get_weights()[1]
#         w3 = model.layers[2].get_weights()[0]
#         b3 = model.layers[2].get_weights()[1]
#         return w1, b1, w2, b2, w3, b3
#     except IndexError:
#         print("Warning: Model architecture might not match expected structure.")
#         return None

# weights = extract_weights_and_biases(model)

# if weights:
#     w1, b1, w2, b2, w3, b3 = weights
    
    
#     folder_name = "weights_folder"
#     os.makedirs(folder_name, exist_ok=True)
    
    
#     np.savetxt(os.path.join(folder_name, 'w1.txt'), w1)
#     np.savetxt(os.path.join(folder_name, 'b1.txt'), b1)
#     np.savetxt(os.path.join(folder_name, 'w2.txt'), w2)
#     np.savetxt(os.path.join(folder_name, 'b2.txt'), b2)
#     np.savetxt(os.path.join(folder_name, 'w3.txt'), w3)
#     np.savetxt(os.path.join(folder_name, 'b3.txt'), b3)
    
#     print("Weights and biases extracted and saved successfully!")
# else:
#     print("Error: Unable to extract weights and biases.")
