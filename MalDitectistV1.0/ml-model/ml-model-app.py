####### MalDitectist ########
## Developed by Nimna Niwarthana ##
## ML Model ##

import pefile
import pandas as pd
import joblib
import os
import math
import tensorflow.keras.models as models

# Load the trained ml models
rf_model = joblib.load('rf_model.joblib')
nn_model = models.load_model('nn_model.h5')

# Load the StandardScaler object used to scale the training data
scaler = joblib.load('scaler.joblib')

# Define a function to extract the required features from the given file
def extract_features(file_path):
    try:
        # Get the file size
        file_length = os.path.getsize(file_path)
        
        # Calculate the entropy of the file
        with open(file_path, 'rb') as f:
            data = bytearray(f.read())
            entropy = 0
            if len(data) > 0:
                # Calculate the frequency of each byte value
                freq_list = []
                for i in range(256):
                    freq_list.append(float(data.count(i))/len(data))
                # Calculate the entropy of the file
                for freq in freq_list:
                    if freq > 0:
                        entropy += -freq * math.log(freq, 2)
        
        # Extract other features from the file
        pe = pefile.PE(file_path)
        features = {
            'machine_type': pe.FILE_HEADER.Machine,
            'number_of_sections': pe.FILE_HEADER.NumberOfSections,
            'timestamp': pe.FILE_HEADER.TimeDateStamp,
            'pointer_to_symbol_table': pe.FILE_HEADER.PointerToSymbolTable,
            'number_of_symbols': pe.FILE_HEADER.NumberOfSymbols,
            'size_of_optional_header': pe.FILE_HEADER.SizeOfOptionalHeader,
            'characteristics': pe.FILE_HEADER.Characteristics,
            'iat_rva': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IAT']].VirtualAddress,
            'major_version': pe.OPTIONAL_HEADER.MajorLinkerVersion,
            'minor_version': pe.OPTIONAL_HEADER.MinorLinkerVersion,
            'check_sum': pe.OPTIONAL_HEADER.CheckSum,
            'compile_date': pe.FILE_HEADER.TimeDateStamp,
            'datadir_IMAGE_DIRECTORY_ENTRY_BASERELOC_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_BASERELOC']].Size,
            'datadir_IMAGE_DIRECTORY_ENTRY_EXPORT_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].Size,
            'datadir_IMAGE_DIRECTORY_ENTRY_IAT_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IAT']].Size,
            'datadir_IMAGE_DIRECTORY_ENTRY_IMPORT_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].Size,
            'debug_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DEBUG']].Size,
            'export_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].Size,
            'size_of_code': pe.OPTIONAL_HEADER.SizeOfCode,
            'size_of_initialized_data': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'size_of_uninitialized_data': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            'size_of_image': pe.OPTIONAL_HEADER.SizeOfImage,
            'size_of_headers': pe.OPTIONAL_HEADER.SizeOfHeaders,
            'subsystem': pe.OPTIONAL_HEADER.Subsystem,
            'major_operating_system_version': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            'minor_operating_system_version': pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            'number_of_rva_and_sizes': pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
            'base_of_code': pe.OPTIONAL_HEADER.BaseOfCode,
            'entry_point_rva': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'resource_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].Size,
            'size_of_heap_commit': pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            'size_of_heap_reserve': pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            'size_of_stack_commit': pe.OPTIONAL_HEADER.SizeOfStackCommit,
            'size_of_stack_reserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
            'status': pe.OPTIONAL_HEADER.DllCharacteristics,
            'file_length': file_length,
            'entropy': entropy,
        }
        return pd.DataFrame(features, index=[0])
    except Exception as e:
        print(f"Error while processing file: {file_path}")
        print(f"Error message: {str(e)}")
        return None


# Define the Predic funtion for NN Model
def predic_nn_model_fn(model, input_data):
    return model(input_data)


# Define a function to classify the given file using the trained model
def classify_file(file_path):
    # Extract features from the file
    file_features = extract_features(file_path)
    if file_features is not None:
        # Scale the features using the same StandardScaler object used to scale the training data
        scaled_features = scaler.transform(file_features.values)
        
        # Make predictions using the trained model - RF Model
        rf_prediction = rf_model.predict(scaled_features)
        proba = rf_model.predict_proba(scaled_features)
        
        # Make predictions using the trained model - NN Model
        nn_prediction = predic_nn_model_fn(nn_model, scaled_features)[0]
        
        # Add prediction values to variables
        rf_pred_value = proba[0][1]
        nn_pred_value = nn_prediction[0]
        
        
        if rf_prediction[0] == 1 and proba[0][1] >= 0.80:
            if nn_prediction[0] >= 0.7:
                print(f"\n{file_path} is predicted as Malware.")
                print(f"Probability Rate of RF Model: {rf_pred_value*100:.2f}%")
                print(f"Prediction of NN Model: {nn_pred_value*100}")
                
            else:
                print(f"\n{file_path} could be a Malware or Legitimate.\
                      \nModels are unable to verify that.\nMostly this is a Malware file.")
                print(f"Probability Rate of RF Model: {rf_pred_value*100:.2f}%")
                print(f"Prediction of NN Model: {nn_pred_value*100}")
                
        else:
            if nn_prediction[0] >= 0.95 and proba[0][1] >= 0.75:
                print(f"\n{file_path} could be a Malware or Legitimate.\
                      \nModels are unable to verify that.\nMostly this is a Malware file.")
                print(f"Probability Rate of RF Model: {rf_pred_value*100:.2f}%")
                print(f"Prediction of NN Model: {nn_pred_value*100}")
                
            else:
                print(f"\n{file_path} is predicted as Legitimate.")
                print(f"Probability Rate of RF Model: {rf_pred_value*100:.2f}%")
                print(f"Prediction of NN Model: {nn_pred_value*100}")
                
    else:
        print(f"Could not extract features for file: {file_path}")


# Call the classify_file function with the file path for analysis
# file_path = '/media/nimna/New Volume1/Malware_Dataset/202275'

# file_path = '/media/nimna/New Volume1/Malware/exe/alaska-hot-carpet-king.exe'
file_path = '/home/nimna/Downloads/Legitimate/exe/VSCodeUserSetup-x64-1.58.2.exe'
classify_file(file_path)


# # Set the directory path
# # directory_path = '/home/nimna/Downloads/Legitimate/exe/'
# directory_path = '/media/nimna/New Volume1/Malware/exe/'

# # Loop through all the files in the directory
# for filename in os.listdir(directory_path):
#     # Get the full path of the file
#     file_path = os.path.join(directory_path, filename)
#     # Call the classify_file function with the file path for analysis
#     classify_file(file_path)

