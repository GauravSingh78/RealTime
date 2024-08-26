Welcome to the Random Forest and Multi-Layer Perceptron (MLP) model application! This tool allows you to perform predictions, generate code, and manage models in various ways. Below is a detailed guide on how to use the different features of this application.

# Random Forest Model
The Random Forest model section provides four main functionalities:
![image](https://github.com/user-attachments/assets/73303eeb-87b8-4b60-980e-24ae66f07d92)

####Predict Using Library
####Manual Prediction
####Verilog Code Generation
####State-wise Prediction
## 1. Predict Using Library
![image](https://github.com/user-attachments/assets/a20c3647-bbfa-446e-b860-78a5d65c6dad)

Purpose: Use a pre-trained Random Forest model to make predictions on a new dataset.
Instructions:
Click on the "Random Forest" button.
Select the "Predict Using Library" option.
Upload the CSV file containing the data you want to predict.
Upload the trained model file (in .pickle format).
Click "Predict" to start the prediction process.
## 2. Manual Prediction
Purpose: Manually use a pre-trained Random Forest model to make predictions on a new dataset.
Instructions:
Click on the "Random Forest" button.
Select the "Manual Prediction" option.
Upload the CSV file containing the data you want to predict.
Upload the trained model file (in .pickle format).
Click "Predict" to start the manual prediction process.
## 3. Verilog Code Generation

Purpose: Generate Verilog code for hardware implementation of the Random Forest model.
![image](https://github.com/user-attachments/assets/90b67b1c-325a-4ced-900e-2eae127d787d)

Instructions:
Click on the "Random Forest" button.
Select the "Verilog Code Generation" option.
Upload the trained model file (in .pickle format).
Click "Extract Feature" to extract feature details and save the text file at the desired location.
Select the saved feature file as input.
Click "Generate Verilog Code" and save the generated Verilog code text file in the desired location.
## 4. State-wise Prediction
Purpose: Perform predictions based on different states or categories in your dataset.
Instructions:
Click on the "Random Forest" button.
Select the "State-wise Prediction" option.
Upload the CSV file containing the data you want to predict.
Choose the specific state for which you want to make predictions.
Click "Predict" to start the prediction process for the selected state.


# Multi-Layer Perceptron (MLP) Model
The MLP model section provides four main functionalities:
![image](https://github.com/user-attachments/assets/986010bd-cc47-4f47-bcb4-d0b96020e9f6)

#### Encode
#### Predict Using Library
#### Manual Prediction
#### Weight Generation

## 1. Encode
Purpose: Encode your dataset using one-hot encoding for use with the MLP model.
![image](https://github.com/user-attachments/assets/28f2f6aa-45f8-4914-b495-b9600755370a)

Instructions:
Click on the "MLP" button.
Select the "Encode" option.
Upload the CSV file you want to encode.
The encoded CSV file will be saved in the mlp_test folder.
## 2. Predict Using Library
Purpose: Use a pre-trained MLP model to make predictions on a new dataset.
![image](https://github.com/user-attachments/assets/e96d510d-58d7-4f30-a675-555a26dc16af)

Instructions:
Click on the "MLP" button.
Select the "Predict Using Library" option.
Upload the CSV file from the mlp_test folder containing the encoded data.
Upload the trained MLP model file (in .h5 format) from the trained_model folder.
Click "Predict" to start the prediction process.
## 3. Manual Prediction
Purpose: Manually use a pre-trained MLP model to make predictions on a new dataset.
Instructions:
Click on the "MLP" button.
Select the "Manual Prediction" option.
Upload the CSV file from the mlp_test folder containing the encoded data.
Upload the trained MLP model file (in .h5 format) from the trained_model folder.
Click "Predict" to start the manual prediction process.
## 4. Weight Generation
Purpose: Generate weights from a trained MLP model.
![image](https://github.com/user-attachments/assets/79e1fe0f-f6a5-420a-9b36-fbedf5329d34)

Instructions:
Click on the "MLP" button.
Select the "Weight Generation" option.
Upload the trained model file (in .h5 format).
Click "Generate Weights" to extract and save the weights in the weight folder.
Additional Notes
Ensure all model files are correctly formatted and saved in the required formats (.pickle for Random Forest models and .h5 for MLP models).
All output files will be saved in their respective folders for easy access.
Follow the instructions carefully for each feature to ensure accurate predictions and results.
