####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Neural Network - Arch01 Model ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset - Used preprocessed dataset that created using data_preprocessing_pe_files.py
dataset = pd.read_csv('../preprocessed_dataset.csv')

# Create dependent & independent variable vectors
x = dataset.iloc[:, :-3].values #independent - pe file features
y = dataset.iloc[:,-1].values #dependent - classification_list (Blacklist = 1 / Whitelist = 0)
num_features = x.shape[1]

# x = dataset.iloc[:, :-1].values  (removed only classification_list )
# x = dataset.iloc[:, :-3].values (removed total_tool_detection, tool_detection_positives, classification_list)

# Split the dataset for training & testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=42)

print("Train Shape:",x_train.shape, "\nTest Shape:", x_test.shape, "\nNum of Features:", num_features)


## Neural Network - Architecture No:01
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Define the model architecture
neuralModelOne = Sequential()
neuralModelOne.add(Dense(1024, input_dim=num_features, activation= 'relu'))
neuralModelOne.add(Dense(512, activation= 'relu'))
neuralModelOne.add(Dense(256, activation= 'relu'))
neuralModelOne.add(Dense(128, activation= 'relu'))
neuralModelOne.add(Dense(64, activation= 'relu'))
neuralModelOne.add(Dense(32, activation= 'relu'))
neuralModelOne.add(Dense(16, activation= 'relu'))
neuralModelOne.add(Dense(8, activation= 'relu'))
neuralModelOne.add(Dense(4, activation= 'relu'))
neuralModelOne.add(Dense(1, activation= 'sigmoid'))

### Print Model Summary
neuralModelOne.summary() 

### Compile Model
neuralModelOne.compile(loss= "binary_crossentropy", optimizer= "rmsprop", metrics=["accuracy"])

### Fit the model and save the trained model in a variable
trained_nn_model_one = neuralModelOne.fit(x_train, y_train, epochs=30, batch_size=64, 
                                          validation_data=(x_test, y_test))



### Model Evaluation
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,r2_score,precision_score

### Accuracy on the train dataset
nmo_y_train_pred = neuralModelOne.predict(x_train)
nmo_y_train_pred = [1 if y>=0.5 else 0 for y in nmo_y_train_pred]

trainAS = round(accuracy_score(y_train, nmo_y_train_pred)*100, 4)

### Accuracy on the test dataset - validation accuracy
nmo_y_test_pred = neuralModelOne.predict(x_test)
nmo_y_test_pred = [1 if y>= 0.5 else 0 for y in nmo_y_test_pred]

testAS = accuracy_score(y_test, nmo_y_test_pred)
val_loss = round((1 - testAS)*100, 4)
val_accurcy = round(testAS*100, 4)


# Calculate precision
precision = round(precision_score(y_test, nmo_y_test_pred)*100, 4)

# Calculate F1 Score
f1 = round(f1_score(y_test, nmo_y_test_pred)*100, 4)

# Calculate R2 score
r2 = round(r2_score(y_test, nmo_y_test_pred)*100, 4)

print("\nAccuracy on the train dataset:", trainAS)
print("Validation Accuracy:", val_accurcy)
print("Validation Loss:", val_loss)
print("Precision:", precision)
print("F1 Score:", f1)
print("R2 Score:", r2)


## Confusion matrix - Neural Network - Architecture No:01
### Confusion matrix without Normalization
nmo_conf_matrix = confusion_matrix(y_test, nmo_y_test_pred)

### Confusion matrix with Normalization
nmo_conf_matrix_norm = confusion_matrix(y_test, nmo_y_test_pred, normalize='true')

### Calculate MSE & MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error

### Calculate the Mean Squared Error (MSE)
nmo_mse = mean_squared_error(y_test, nmo_y_test_pred)

### Calculate the Mean Absolute Error (MAE)
nmo_mae = mean_absolute_error(y_test, nmo_y_test_pred)

print("\nMean Squard Error:", nmo_mse)
print("Mean Absolute Error:", nmo_mae)

# Calculate TP, TN, FP, FN
TP = round(nmo_conf_matrix_norm[1, 1]*100, 4)
TN = round(nmo_conf_matrix_norm[0, 0]*100, 4)
FP = round(nmo_conf_matrix_norm[0, 1]*100, 4)
FN = round(nmo_conf_matrix_norm[1, 0]*100, 4)

print("\nTrue Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)



# Plot the training and validation accuracy over the epochs
plt.plot(trained_nn_model_one.history['accuracy'])
plt.plot(trained_nn_model_one.history['val_accuracy'])
plt.title('NNA 01 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot the training and validation loss over the epochs
plt.plot(trained_nn_model_one.history['loss'])
plt.plot(trained_nn_model_one.history['val_loss'])
plt.title('NNA 01 Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# save the model
neuralModelOne.save('nn_model.h5')

## ---------- ##