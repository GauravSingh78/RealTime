import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pickle
import numpy as np

def generate_plots(csv_file_path, pickle_file_path):
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
    prediction = loaded_model.predict(new_df)

    # Generate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot input waveform
    ax1.plot(data['time'], data['vinp'], label='Input Vinp', color='blue')
    ax1.plot(data['time'], data['vdd'], label='Input Vdd', color='red')
    ax1.plot(data['time'], data['pd'], label='Input Pd', color='green')
    ax1.plot(data['time'], data['temperature'], label='Input Temperature', color='orange')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Input Waveform')
    ax1.legend()

    # Plot output waveform
    ax2.plot(data['time'], data['vinn'], label='Actual Vinn', color='blue')
    ax2.plot(data['time'], prediction, label='Predicted Vinn', color='red')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.set_title('Output Waveform')
    ax2.legend()

    # Save plot image to bytes buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_bytes = buf.read()
    buf.close()

    # Convert bytes to base64-encoded string
    plot_base64 = base64.b64encode(plot_bytes).decode('utf-8')

    return plot_base64

if __name__ == "__main__":
    csv_file_path = input("Enter the path to the CSV file: ")
    pickle_file_path = input("Enter the path to the pickle file: ")
    generate_plots(csv_file_path, pickle_file_path)



# import pandas as pd
# import matplotlib.pyplot as plt
# import base64
# from io import BytesIO
# import pickle
# import numpy as np
# from itertools import count
# from matplotlib.animation import FuncAnimation

# def generate_plots(csv_file_path, pickle_file_path):
#     # Read CSV file
#     data = pd.read_csv(csv_file_path)

#     # Load the trained model
#     with open(pickle_file_path, 'rb') as file:
#         model_data = pickle.load(file)

#     loaded_model = model_data['model']
#     label_encoder = model_data['label_encoder']

#     new_data = data[['vdd', 'pd', 'vinp', 'temperature', 'process']]
#     new_df = pd.DataFrame(new_data, columns=['vdd', 'pd', 'vinp', 'temperature', 'process'])
#     new_df['process'] = label_encoder.transform(new_df['process'])
#     prediction = loaded_model.predict(new_df)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

#     x_data_input = []
#     pdd = []
#     vdd = []
#     vinp = []
#     temperature = []
#     process = []

#     index = count()

#     def animate_input(i):
#         if next(index) < len(data):
#             x = data['time'].iloc[next(index)]
#             p = data['pd'].iloc[next(index)]
#             vin = data['vinp'].iloc[next(index)]
#             vd = data['vdd'].iloc[next(index)]
#             temp = data['temperature'].iloc[next(index)]
#             pro = data['process'].iloc[next(index)]

#             x_data_input.append(x)
#             pdd.append(p)
#             vinp.append(vin)
#             vdd.append(vd)
#             temperature.append(temp)
#             process.append(pro)

#             ax1.clear()
#             ax1.plot(pdd, label='pdd')
#             ax1.plot(vinp, label='vinp')
#             ax1.plot(vdd, label='vdd')
#             ax1.plot(temperature, label='temperature')
#             ax1.plot(process, label='process')
#             ax1.set_xlabel('Time')
#             ax1.set_ylabel('Voltage')
#             ax1.set_title('Real-time Waveform Plot (Input)')
#             ax1.legend()

#     x_data_output = []
#     y_data = []
#     vinn = []

#     def animate_output(i):
#         if next(index) < len(data):
#             x = data['time'].iloc[next(index)]
#             y = data['vinn'].iloc[next(index)]
#             vin = prediction[next(index)]

#             x_data_output.append(x)
#             y_data.append(y)
#             vinn.append(vin)

#             ax2.clear()
#             ax2.plot(y_data, label='Actual', color='b')
#             ax2.plot(vinn, label='Predicted', color='red')
#             ax2.set_xlabel('Time')
#             ax2.set_ylabel('Voltage')
#             ax2.set_title('Real-time Waveform Plot (Output)')
#             ax2.legend()

#     ani_input = FuncAnimation(fig, animate_input, interval=50, cache_frame_data=False)
#     ani_output = FuncAnimation(fig, animate_output, interval=50, cache_frame_data=False)

#     # Save plot image to bytes buffer
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plot_bytes = buf.read()
#     buf.close()

#     # Convert bytes to base64-encoded string
#     plot_base64 = base64.b64encode(plot_bytes).decode('utf-8')

#     return plot_base64

# if __name__ == "__main__":
#     csv_file_path = input("Enter the path to the CSV file: ")
#     pickle_file_path = input("Enter the path to the pickle file: ")
#     generate_plots(csv_file_path, pickle_file_path)
