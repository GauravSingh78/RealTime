import json
import time
from flask import Flask, flash, jsonify, render_template, request, redirect, url_for
import os
from flask_socketio import SocketIO, emit
from matplotlib import pyplot as plt
from VerilogCode import generate_and_save_verilog
from VerilogRule import extract_and_save_rules
import pickle
from predictionrf import generate_plots
from io import BytesIO
import base64
from werkzeug.utils import secure_filename
from predictionrfman import generate_man_plots
from io import BytesIO
import base64
import pandas as pd
from sklearn.tree import _tree
# from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model  
from tensorflow.keras.metrics import MeanSquaredError # type: ignore
import numpy as np
from weightExtract import extract_and_save_weights
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog



plt.switch_backend('agg')




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/random_forest', methods=['GET', 'POST'])
def random_forest():
    if request.method == 'POST':
        option = request.form['option']
        if option == 'predict_library':
            return redirect(url_for('prediction'))
        elif option == 'manual_prediction':
            return redirect(url_for('manPrediction'))
        elif option == 'verilog_code':
            return redirect(url_for('verilog_generation'))
        elif option == 'regions_predict':
            return redirect(url_for('regions'))
    return render_template('randomForest.html')

@app.route('/mlp', methods=['GET', 'POST'])
def mlp():
    if request.method == 'POST':
        option = request.form['option']
        if option == 'predict_library':
            return redirect(url_for('mlpPrediction'))
        elif option == 'encode_file':
            return redirect(url_for('encode'))
        elif option == 'manual_prediction':
            return redirect(url_for('mlpManPrediction'))
        elif option == 'generate_weight':
            return redirect(url_for('weightGeneration'))
    return render_template('mlp.html')

@app.route('/FeatureExtraction', methods=['POST'])
def feature_extraction():
    if request.method == 'POST':
        if 'pickle_file' not in request.files:
            return 'No file part'

        file = request.files['pickle_file']

        if file.filename == '':
            return 'No selected file'

        if file:
            try:
                
                filename = secure_filename(file.filename)
                pickle_file_path = os.path.join('trained_model', filename)
                file.save(pickle_file_path)

                
                # extract_and_save_rules(pickle_file_path, 'RulesFolder')
                extract_and_save_rules(pickle_file_path)


                message ="If-Else Rule generated Successfully ... "
                return render_template('success.html', message=message)
            except Exception as e:
                return f'An error occurred: {str(e)}'

    return render_template('verilog.html')




def ask_folder_and_filename():
    # Create a Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask user to choose the file path for saving
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", title="Save Verilog Module As", filetypes=[("Text files", "*.txt")])
    if not save_path:
        print("No file saved. Exiting.")
        return None, None

    output_folder = os.path.dirname(save_path)
    filename = os.path.basename(save_path)

    return output_folder, filename

@app.route('/RuleGeneration', methods=['POST'])
def rule_generation():
    if request.method == 'POST':
        
        if 'input_file' not in request.files:
            return 'No file part'
        file = request.files['input_file']
        
        if file.filename == '':
            return 'No selected file'
        if file:
            
            text_file_path =  os.path.join('RulesFolder',file.filename)
            file.save(text_file_path)
            output_folder, _ = ask_folder_and_filename()
            # generate_and_save_verilog(text_file_path, 'VerilogCodeFolder')
            generate_and_save_verilog(text_file_path, output_folder)

            
            message ="Verilog Code Generated Successfully ..."
            return render_template('success.html', message=message)

    return render_template('verilog.html')


@app.route('/verilog_generation', methods=['GET'])
def verilog_generation():
 
    
    return render_template('verilog.html')


@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html')

data = None
model = None
label_encoder = None


@app.route('/predict', methods=['GET','POST'])
def predict():
        global data, model, label_encoder
        if request.method == 'POST':

            if 'csv_file' not in request.files or 'pickle_file' not in request.files:
                return 'Please upload both CSV file and trained model.'

            csv_file = request.files['csv_file']
            pickle_file = request.files['pickle_file']

            
            csv_file_path = os.path.join('test', csv_file.filename)
            pickle_file_path = os.path.join('trained_model', pickle_file.filename)
            csv_file.save(csv_file_path)    
            pickle_file.save(pickle_file_path)

            
            plot_base64 = generate_plots(csv_file_path, pickle_file_path)
        

            data = pd.read_csv(csv_file_path)
            with open(pickle_file_path, 'rb') as file:
                model_data = pickle.load(file)
            model=model_data['model']
            label_encoder=model_data['label_encoder']

            new_data = data[['vdd', 'pd', 'vinp', 'temperature','process']]
            new_df = pd.DataFrame(new_data, columns=['vdd', 'pd', 'vinp', 'temperature','process'])
            new_df['process'] = label_encoder.transform(new_df['process'])
        
            prediction = model.predict(new_df)

         
        
             
            time = data['time'].tolist()
            vinn_actual = data['vinn'].tolist()
            vinn_predicted = prediction.tolist()

            return render_template('wavefrom_output.html', time=time, vinn_actual=vinn_actual, vinn_predicted=vinn_predicted)

        return render_template('prediction.html')
        



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
        if feature == -2:  
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


@app.route('/manPrediction', methods=['GET','POST'])
def manPrediction():
    return render_template('manPrediction.html')

data = None
model = None
label_encoder = None

@app.route('/manPredict', methods=['GET','POST'])
def manPredict():
        
        if request.method == 'POST':

            if 'csv_file' not in request.files or 'pickle_file' not in request.files:
                return 'Please upload both CSV file and trained model.'

            csv_file = request.files['csv_file']
            pickle_file = request.files['pickle_file']

            
            csv_file_path = os.path.join('test', csv_file.filename)
            pickle_file_path = os.path.join('trained_model', pickle_file.filename)
            csv_file.save(csv_file_path)    
            pickle_file.save(pickle_file_path)

            
            plot_base64 = generate_man_plots(csv_file_path, pickle_file_path)
        

            data = pd.read_csv(csv_file_path)
            with open(pickle_file_path, 'rb') as file:
                model_data = pickle.load(file)
            model=model_data['model']
            label_encoder=model_data['label_encoder']

            new_data = data[['vdd', 'pd', 'vinp', 'temperature','process']]
            feature = ['vdd', 'pd', 'vinp', 'temperature','process']
            new_df = pd.DataFrame(new_data, columns=['vdd', 'pd', 'vinp', 'temperature','process'])
            new_df['process'] = label_encoder.transform(new_df['process'])
            predictions = []
            for index, row in new_df.iterrows():
                
                prediction = manual_predict(model, row.values)
                predictions.append(prediction)
            
        
             
            time = data['time'].tolist()
            vinn_actual = data['vinn'].tolist()
            vinn_predicted = predictions

            return render_template('wavefrom_output.html', time=time, vinn_actual=vinn_actual, vinn_predicted=vinn_predicted)

        return render_template('manPrediction.html')
       






# MLP Server



@app.route('/mlpPrediction', methods=['GET'])
def mlpPrediction():
    return render_template('mlpPrediction.html')


@app.route('/mlpPredict', methods=['GET','POST'])
def mlpPredict():
        global data, model, label_encoder
        if request.method == 'POST':

            if 'csv_file' not in request.files or 'hdf_file' not in request.files:
                return 'Please upload both CSV file and trained model.'

            csv_file = request.files['csv_file']
            hdf_file = request.files['hdf_file']

            csv_file_path = os.path.join('mlpTest', csv_file.filename)
            hdf_file_path = os.path.join('trained_model', hdf_file.filename)

            # inputs =['vinp', 'pd', 'vdd','temperature','process_fastnfastp','process_fastnslowp', 'process_slownfastp', 'process_slownslowp','process_typical','vinn']
            data =pd.read_csv(csv_file_path)
            # print("Head===========")
            # print(data.head)
            # new_data = data[inputs]
            # X_test = new_data.drop('vinn', axis=1).values
            # # X_test = X_test.astype(np.float32)
            # # Make predictions
            # # y_pred = model.predict(X_test)

            # # new_df = pd.DataFrame(new_data, columns=['vinp', 'pd', 'vdd','temperature','process_fastnfastp','process_fastnslowp', 'process_slownfastp', 'process_slownslowp','process_typical'])
            # # new_df = pd.DataFrame(X_test)
            # print("=====================",hdf_file_path)
            # vinn = data['vinn'].values
            model = load_model(hdf_file_path)
            print(model)
            
            # predictions= model_loaded.predict(X_test)
            #             # Check the type of predictions
            # print("Type of predictions:", type(predictions))

            

            # predictions_list = predictions.tolist()
            inputs = ['vinp', 'xpd', 'vdd', 'temperature', 'process_fastnfastp', 'process_fastnslowp', 
          'process_slownfastp', 'process_slownslowp', 'process_typical', 'vinn']
            df_test1 = data[inputs]
            print(data.head)
            # Prepare the input features for prediction
            X_test = df_test1.drop('vinn', axis=1)

            # Make predictions
            y_pred = model.predict(X_test)

            # Print predictions
            print("Predictions:")
            print(y_pred)

            # # Check the type of predictions_list
            # print("Type of predictions_list:", type(predictions_list))
            
            # # Check the shape of predictions_list
            # if len(predictions_list) > 0:
            #     shape = (len(predictions_list), len(predictions_list[0]))
            #     print("Shape of predictions_list:", shape)
            # else:
            #     print("predictions_list is empty.")


            
            # # Convert predictions_list to numpy array
            # predictions_array = np.array(predictions_list)

            # # Reshape predictions_array
            # predictions_array = predictions_array.reshape(-1)

            # # Check the shape of predictions_array
            # print("Shape of predictions_array:", predictions_array.shape)

            # print("++++++++++++++++++++++++",predictions_array)

            # print("+++++++++++++++++++++++++++++++++++++++++++++++++")
            # print(y_pred)
            
            time = data['time'].tolist()
            vinn_actual = data['vinn'].tolist()
            vinn_predicted = y_pred.tolist()
            predictions_flat = [item for sublist in vinn_predicted for item in sublist]
            
           

            # PLOT

            # plt.plot(time,vinn_actual,label='Actual',color='blue')
            # plt.plot(time,predictions_flat,label="Predicted", color='red')
            # plt.title("Dot Predict Prediction")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

            

            return render_template('wavefrom_output.html', time=time, vinn_actual=vinn_actual, vinn_predicted=predictions_flat)

        return render_template('mlpPrediction.html')
        


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
    

def forward_propagation(inputs, weights):
    
    w1, b1, w2, b2, w3, b3 = weights
    b1_reshaped = b1.reshape(1, -1)
    b2_reshaped = b2.reshape(1, -1)
    b3_reshaped = b3.reshape(1, -1)
    z1 = np.dot(inputs, w1) + b1_reshaped
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2_reshaped
    a2 = relu(z2)
    z3 = np.dot(a2, w3) + b3_reshaped
    
    return z3
def relu(x):
    return np.maximum(0, x)


@app.route('/mlpManPrediction', methods=['GET','POST'])
def mlpManPrediction():
    return render_template('mlpManPrediction.html')


@app.route('/mlpManPredict', methods=['GET','POST'])
def mlpManPredict():
        
        if request.method == 'POST':

            if 'csv_file' not in request.files or 'hdf_file' not in request.files:
                return 'Please upload both CSV file and trained model.'

            csv_file = request.files['csv_file']
            hdf_file = request.files['hdf_file']

            csv_file_path = os.path.join('mlpTest', csv_file.filename)
            hdf_file_path = os.path.join('trained_model', hdf_file.filename)

            inputs =['vinp', 'pd', 'vdd','temperature','process_fastnfastp','process_fastnslowp', 'process_slownfastp', 'process_slownslowp','process_typical','vinn']
            data =pd.read_csv(csv_file_path)
            new_data = data[inputs]
            # new_df = pd.DataFrame(new_data, columns=['vinp', 'pd', 'vdd','temperature','process_fastnfastp','process_fastnslowp', 'process_slownfastp', 'process_slownslowp','process_typical'])
            new_df = new_data.drop('vinn', axis=1).values
            
            print(data.head)
            model = load_model(hdf_file_path, custom_objects={'MeanSquaredError': MeanSquaredError, 'mse': 'mean_squared_error'})
            print(model)
            weights = extract_weights_and_biases(model)
            predictions = forward_propagation(new_df, weights)

            print(predictions)
            
             
            time = data['time'].tolist()
            vinn_actual = data['vinn'].tolist()
            vinn_predicted = predictions.tolist()
            predictions_flat = [item for sublist in vinn_predicted for item in sublist]
            
            return render_template('wavefrom_output.html', time=time, vinn_actual=vinn_actual, vinn_predicted=predictions_flat)

        return render_template('mlpManPrediction.html')
        

app.secret_key = "super secret key"

UPLOAD_FOLDER = 'test'
ENCODING_FOLDER = 'mlpTest'


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(ENCODING_FOLDER):
    os.makedirs(ENCODING_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/encode', methods=['GET','POST'])
def encode():
    if request.method == 'POST':
        
        if 'csv_file' not in request.files:
            flash('Please upload both CSV file and trained model.', 'error')
            return render_template('encode.html')

        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            flash('No selected file', 'error')
            return render_template('encode.html')

        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)
        csv_file.save(csv_file_path)
        print("name=============== ======",csv_file.filename)
        df = pd.read_csv(csv_file_path)

        
        df['process'] = df['process'].astype('category')
        df_train = pd.get_dummies(df, columns=['process'])

        
        df_train.replace({True: 1}, inplace=True)

        if df_train is not None:
           
            df_train.drop(columns=['process'], inplace=True, errors='ignore')

            all_processes = {'process_fastnfastp', 'process_fastnslowp', 'process_slownfastp', 'process_slownslowp', 'process_typical'}
            for process_col in all_processes:
                if process_col not in df_train.columns:
                    df_train[process_col] = 0

            ordered_columns = ['process_fastnfastp', 'process_fastnslowp', 'process_slownfastp', 'process_slownslowp', 'process_typical']
            df_train = df_train.reindex(columns=list(df.columns) + ordered_columns, fill_value=0)


            encoded_csv_path = os.path.join(ENCODING_FOLDER,csv_file.filename )
            df_train.to_csv(encoded_csv_path, index=False)

            message ="File encoded successfully and saved in mlpTest folder."
            return render_template('success.html', message=message)
        else:
            flash('Error occurred while encoding the file.', 'error')
            return render_template('encode.html')





    return render_template('encode.html')




@app.route('/weightGeneration', methods=['GET','POST'])
def weightGeneration():
    if request.method == 'POST':
        
        if 'hdf_file' not in request.files:
            return 'No file part'
        file = request.files['hdf_file']
        
        if file.filename == '':
            return 'No selected file'
        if file:
            
            hdf_file_path =  os.path.join('trained_model',file.filename)
            file.save(hdf_file_path)
            print("+++++++++++++++++++++++",hdf_file_path)
            weights =extract_and_save_weights(hdf_file_path)
            print(weights)
            if not weights:
                print("Error: Unable to extract weights and biases.")
           
            message ="Weights generated successfully and saved in mlp_saved_weights folder..."
            return render_template('success.html', message=message)

    return render_template('weight.html')




################################################
#####  Regions 


import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
s1 =[]
s2 =[]
s3 =[]
s4 =[]
a1=[]
a2=[]
a3=[]
a4=[]

with open("./trained_model/Regions_1.pkl","rb") as file:
    m1 = pickle.load(file)

models1 = m1['model']
feature_names = m1['feature_names']
label_encoder = m1['label_encoder']

with open("./trained_model/Regions_2.pkl","rb") as file:
    m2 = pickle.load(file)

models2 = m2['model']

with open("./trained_model/Regions_3.pkl","rb") as file:
    m3 = pickle.load(file)

models3 = m3['model']

with open("./trained_model/Regions_4.pkl","rb") as file:
    m4 = pickle.load(file)

models4 = m4['model']





plot_folder = os.path.join('static', 'plot')

@app.route('/regions', methods=['GET', 'POST'])
def regions():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return 'No file part'

        file = request.files['csv_file']
        region = request.form['region']
        csv_file_path = os.path.join('test', file.filename)

        data = pd.read_csv(csv_file_path)
        new_data = data[['vdd', 'pd', 'vinp', 'temperature', 'process']]
        new_df = pd.DataFrame(new_data, columns=['vdd', 'pd', 'vinp', 'temperature', 'process'])
        new_df['process'] = label_encoder.transform(new_df['process'])

        if file.filename == '':
            return 'No selected file'

        s, a = [], []
        if region == "region_1":
            for index, row in new_df.iterrows():
                if row['pd'] != 0 and row['vinp'] == 0:
                    prediction = models1.predict(row[feature_names].values.reshape(1, -1))
                    s.append(prediction)
                    a.append(data.at[index, 'vinn'])
            plot_filename = "s1.png"
            plot_title = "Plot of State 1"
            

        elif region == "region_2":
            for index, row in new_df.iterrows():
                if row['pd'] == 0 and row['vinp'] == 0:
                    prediction = models2.predict(row[feature_names].values.reshape(1, -1))
                    s.append(prediction)
                    a.append(data.at[index, 'vinn'])
            plot_filename = "s2.png"
            plot_title = "Plot of State 2"

        elif region == "region_3":
            for index, row in new_df.iterrows():
                if row['pd'] == 0 and row['vinp'] != 0:
                    prediction = models3.predict(row[feature_names].values.reshape(1, -1))
                    s.append(prediction)
                    a.append(data.at[index, 'vinn'])
            plot_filename = "s3.png"
            plot_title = "Plot of State 3"

        elif region == "region_4":
            for index, row in new_df.iterrows():
                if row['pd'] > 0 and row['vinp'] > 0:
                    prediction = models4.predict(row[feature_names].values.reshape(1, -1))
                    s.append(prediction)
                    a.append(data.at[index, 'vinn'])
            plot_filename = "s4.png"
            plot_title = "Plot of State 4"

        plt.plot(a, color='b', label="Actual")
        plt.plot(s, color='red', label="Predicted")
        plt.title(plot_title)
        plt.xlabel("X-axis label")
        plt.ylabel("Y-axis label")
        plt.legend()
        plot_path = os.path.join(plot_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        message = "Region Plotted........"
        return render_template('stateSuccess.html', message=message, plot=plot_filename)

    return render_template('regions.html')




# plot_folder = "plot_folder"
# if not os.path.exists(plot_folder):
#     os.makedirs(plot_folder)


# @app.route('/regions', methods=['GET','POST'])
# def regions():
#     if request.method == 'POST':
        
#         if 'csv_file' not in request.files:
#             return 'No file part'
        

#         file = request.files['csv_file']
#         region = request.form['region']
#         csv_file_path = os.path.join('test', file.filename)

#         data = pd.read_csv(csv_file_path)
#         new_data = data[['vdd', 'pd', 'vinp', 'temperature','process']]
#         new_df = pd.DataFrame(new_data, columns=['vdd', 'pd', 'vinp', 'temperature','process'])
#         new_df['process'] = label_encoder.transform(new_df['process'])


#         if file.filename == '':
#             return 'No selected file'
#         if file :
#             if region =="region_1":
#                 for index, row in new_df.iterrows():
#                     if row['pd'] != 0 and row['vinp'] == 0:
#                         prediction = models1.predict(row[feature_names].values.reshape(1, -1))
#                         s1.append(prediction)
#                         a1.append(data.at[index, 'vinn'])
                
#                 plt.plot(a1, color='b',label="Actual")
#                 plt.plot(s1, color='red',label="Predicted")
#                 plt.title("Plot of State 1")
#                 plt.xlabel("X-axis label")
#                 plt.ylabel("Y-axis label")
#                 plt.legend()
#                 plot_path=os.path.join(plot_folder, "s1.png")
#                 plt.savefig(plot_path)
#                 plt.close()

                

#             elif region =="region_2":
#                 for index, row in new_df.iterrows():
#                     if row['pd'] == 0 and row['vinp'] == 0:
#                         prediction = models2.predict(row[feature_names].values.reshape(1, -1))
#                         s2.append(prediction)
#                         a2.append(data.at[index, 'vinn'])

#                 plt.plot(a2, color='b',label="Actual")
#                 plt.plot(s2, color='red',label="Predicted")
#                 plt.title("Plot of State 2")
#                 plt.xlabel("X-axis label")
#                 plt.ylabel("Y-axis label")
#                 plt.legend()
#                 plot_path=os.path.join(plot_folder, "s2.png")
#                 plt.savefig(plot_path)
#                 plt.close()
                

#             elif region =="region_3":
#                 for index, row in new_df.iterrows():
#                     if row['pd'] == 0 and row['vinp'] != 0:
#                         prediction = models3.predict(row[feature_names].values.reshape(1, -1))
#                         s3.append(prediction)
#                         a3.append(data.at[index, 'vinn'])

                
#                 plt.plot(a3, color='b',label="Actual")
#                 plt.plot(s3, color='red',label="Predicted")
#                 plt.title("Plot of State 3")
#                 plt.xlabel("X-axis label")
#                 plt.ylabel("Y-axis label")
#                 plt.legend()
#                 plot_path=os.path.join(plot_folder, "s3.png")
#                 plt.savefig(plot_path)
#                 plt.close()
                

#             elif region =="region_4":
#                 for index, row in new_df.iterrows():
#                     if row['pd'] > 0 and row['vinp'] > 0:
#                         prediction = models4.predict(row[feature_names].values.reshape(1, -1))
#                         s4.append(prediction)
#                         a4.append(data.at[index, 'vinn'])

#                 plt.plot(a4, color='b',label="Actual")
#                 plt.plot(s4, color='red',label="Predicted")
#                 plt.title("Plot of State 4")
#                 plt.xlabel("X-axis label")
#                 plt.ylabel("Y-axis label")
#                 plt.legend()
#                 plot_path=os.path.join(plot_folder, "s4.png")
#                 plt.savefig(plot_path)
#                 plt.close()

#             message = "Region Plotted........"
#             return render_template('stateSuccess.html', message=message)

#     return render_template('regions.html')


if __name__ == '__main__':
    app.run(debug=True)
