####### MalDitectist ########
## Developed by Nimna Niwarthana ##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv('../dataset1.csv')

x = dataset.iloc[:,4:-4].values
y = dataset.iloc[:,-3].values 


# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset for training & testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

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



## Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
logModel = lr.fit(x_train, y_train) 

### Model Evaluation
### Accuracy on the train dataset
lr_y_train_pred = logModel.predict(x_train)
accuracy_score(y_train, lr_y_train_pred)

### Accuracy on the test dataset
lr_y_test_pred = logModel.predict(x_test)
accuracy_score(y_test, lr_y_test_pred)

f1_score(y_test, lr_y_test_pred)



## Confusion matrix - Logistic Regression
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



## Neural Network - Architecture No:01
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### Define the model architecture
neuralModelOne = Sequential()
neuralModelOne.add(Dense(16, input_dim=4, activation= 'relu'))
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



## Neural Network - Architecture No:02







