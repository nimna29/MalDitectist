####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Model Building 1 - Supervised Learning v 1.0.0 ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset - Used preprocessed dataset that created using data_preprocessing_pe_files.py
dataset = pd.read_csv('../preprocessed_dataset.csv')

# Create dependent & independent variable vectors
x = dataset.iloc[:, :-1].values #independent - other features (removed classification_list)
y = dataset.iloc[:,-1].values #dependent - classification_list (Blacklist = 1 / Whitelist = 0)

# Split the dataset for training & testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=42)

print(x_train.shape,x_test.shape)



# Model Building - Supervised Learning

## Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rfc = RandomForestClassifier(max_depth=2, random_state=0)
randomModel = rfc.fit(x_train, y_train)

### Model Evaluation
from sklearn.metrics import f1_score,accuracy_score,plot_confusion_matrix,confusion_matrix

### Accuracy on the train dataset
rf_y_train_pred = randomModel.predict(x_train)
accuracy_score(y_train, rf_y_train_pred)

### Accuracy on the test dataset
rf_y_test_pred = randomModel.predict(x_test)
accuracy_score(y_test, rf_y_test_pred)

f1_score(y_test, rf_y_test_pred)



## Confusion matrix - Random Forest
### Confusion matrix without Normalization
conf_matrix = confusion_matrix(y_test, rf_y_test_pred)

### Confusion matrix with Normalization
conf_matrix_norm = confusion_matrix(y_test, rf_y_test_pred, normalize='true')

### Create a figure iwth two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

### Plot the 1st confusion matrix 
plot_confusion_matrix(randomModel, x_test, y_test,
                      cmap=plt.cm.Blues, ax=ax1)
ax1.set_title("Confusion Matrix")

### Plot the 2nd confusion matrix (Normalized)
plot_confusion_matrix(randomModel, x_test, y_test, 
                      normalize='true',
                      cmap=plt.cm.Blues, ax=ax2)
ax2.set_title("CM (Normalized)")

### Add top title and ahow
plt.suptitle("Confustion Matrices - Random Forest")
plt.subplots_adjust(wspace=0.6)
plt.show()
## ---------- ##


## Logistic Regression - Default: lbfgs
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0, max_iter=1000)
logModel = lr.fit(x_train, y_train) 

### Model Evaluation
### Accuracy on the train dataset
lr_y_train_pred = logModel.predict(x_train)
accuracy_score(y_train, lr_y_train_pred)

### Accuracy on the test dataset
lr_y_test_pred = logModel.predict(x_test)
accuracy_score(y_test, lr_y_test_pred)

f1_score(y_test, lr_y_test_pred)


## Confusion matrix - Logistic Regression - Default: lbfgs
### Confusion matrix without Normalization
lr_conf_matrix = confusion_matrix(y_test, lr_y_test_pred)

### Confusion matrix with Normalization
lr_conf_matrix_norm = confusion_matrix(y_test, lr_y_test_pred, normalize='true')

### Create a figure iwth two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

### Plot the 1st confusion matrix 
plot_confusion_matrix(logModel, x_test, y_test,
                      cmap=plt.cm.Blues, ax=ax1)
ax1.set_title("Confusion Matrix")

### Plot the 2nd confusion matrix (Normalized)
plot_confusion_matrix(logModel, x_test, y_test, 
                      normalize='true',
                      cmap=plt.cm.Blues, ax=ax2)
ax2.set_title("CM (Normalized)")

### Add top title and ahow
plt.suptitle("Confustion Matrices - Logistic Regression")
plt.subplots_adjust(wspace=0.6)
plt.show()
## ---------- ##


## Logistic Regression - Solver: saga
lrsaga = LogisticRegression(random_state=0, solver='saga', max_iter=10000)
logSagaModel = lrsaga.fit(x_train, y_train) 

### Model Evaluation
### Accuracy on the train dataset
lrsaga_y_train_pred = logSagaModel.predict(x_train)
accuracy_score(y_train, lrsaga_y_train_pred)

### Accuracy on the test dataset
lrsaga_y_test_pred = logSagaModel.predict(x_test)
accuracy_score(y_test, lrsaga_y_test_pred)

f1_score(y_test, lrsaga_y_test_pred)


## Confusion matrix - Logistic Regression - Solver: saga
### Confusion matrix without Normalization
lrsaga_conf_matrix = confusion_matrix(y_test, lrsaga_y_test_pred)

### Confusion matrix with Normalization
lrsaga_conf_matrix_norm = confusion_matrix(y_test, lrsaga_y_test_pred, normalize='true')

### Create a figure iwth two subplots
fig, (ax1, ax2) = plt.subplots(1, 2)

### Plot the 1st confusion matrix 
plot_confusion_matrix(logSagaModel, x_test, y_test,
                      cmap=plt.cm.Blues, ax=ax1)
ax1.set_title("Confusion Matrix")

### Plot the 2nd confusion matrix (Normalized)
plot_confusion_matrix(logSagaModel, x_test, y_test, 
                      normalize='true',
                      cmap=plt.cm.Blues, ax=ax2)
ax2.set_title("CM (Normalized)")

### Add top title and ahow
plt.suptitle("Confustion Matrices - Logistic Regression: saga")
plt.subplots_adjust(wspace=0.6)
plt.show()
## ---------- ##


## Neural Network - Architecture No:01
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Define the model architecture
neuralModelOne = Sequential()
neuralModelOne.add(Dense(16, input_dim=39, activation= 'relu'))
neuralModelOne.add(Dense(8, activation= 'relu'))
neuralModelOne.add(Dense(4, activation= 'relu'))
neuralModelOne.add(Dense(1, activation= 'sigmoid'))

### Print Model Summary
neuralModelOne.summary() 

### Compile Model
neuralModelOne.compile(loss= "binary_crossentropy", optimizer= "rmsprop", metrics=["accuracy"])

### Fit Model
neuralModelOne.fit(x_train, y_train, epochs=5, batch_size=32)


### Model Evaluation
### Accuracy on the train dataset
nmo_y_train_pred = neuralModelOne.predict(x_train)
nmo_y_train_pred = [1 if y>=0.5 else 0 for y in nmo_y_train_pred]

accuracy_score(y_train, nmo_y_train_pred)

### Accuracy on the test dataset
nmo_y_test_pred = neuralModelOne.predict(x_test)
nmo_y_test_pred = [1 if y>= 0.5 else 0 for y in nmo_y_test_pred]

accuracy_score(y_test, nmo_y_test_pred)

f1_score(y_test, nmo_y_test_pred)


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

print("Mean Squard Error:", nmo_mse)
print("Mean Absolute Error:", nmo_mae)


# Fit the model and save the training history
history = neuralModelOne.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Plot the training and validation accuracy over the epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('NNA 01 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Plot the training and validation loss over the epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('NNA 01 Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
## ---------- ##



## Neural Network - Architecture No:02 - TBA

