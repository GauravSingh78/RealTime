import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pickle
from sklearn.tree import _tree
import numpy as np

def manual_predict(model, features):
    predictions = []
    for tree in model.estimators_:
        tree_prediction = traverse_tree(tree.tree_, features)
        predictions.append(tree_prediction)
    return sum(predictions) / len(predictions)

def traverse_tree(tree, features):
    node_id = 0
    while True:
        feature = tree.feature[node_id]
        if feature == -2:  # Leaf node
            return tree.value[node_id][0][0]
        else:
            if features[feature] <= tree.threshold[node_id]:
                next_node_id = tree.children_left[node_id]
            else:
                next_node_id = tree.children_right[node_id]
            if next_node_id == -1:
                return tree.value[node_id][0][0]
            else:
                node_id = next_node_id

def generate_man_plots(csv_file_path, pickle_file_path):
    # Read CSV file
    data = pd.read_csv(csv_file_path)

    # Load the trained model
    with open(pickle_file_path, 'rb') as file:
        model_data = pickle.load(file)

    loaded_model = model_data['model']
    label_encoder = model_data['label_encoder']

    new_data = data[['vdd', 'pd', 'vinp', 'temperature', 'process']]
    new_df = pd.DataFrame(new_data, columns=['vdd', 'pd', 'vinp', 'temperature', 'process'])
    new_df['process'] = label_encoder.transform(new_df['process'])

    # Predict using loaded_model.predict
    prediction = loaded_model.predict(new_df)
    
   
    
    
    # Plot output waveform
    # plt.plot( data['vinn'], label='Actual Vinn', color='blue')
    # plt.plot( prediction, label='Predicted Vinn', color='red')
    
    # plt.title("RF Manual Prediction")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()



csv_file_path = "./test/slownfastp_3.6V_45.csv"
pickle_file_path = "./trained_model/Random_Forest_20.pkl"
generate_man_plots(csv_file_path, pickle_file_path)
