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



# Model Building
## Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

rfc = RandomForestClassifier(max_depth=2, random_state=0)
randomModel = rfc.fit(x_train, y_train)

### Random Forest - Evaluation on test data
from sklearn.metrics import f1_score,accuracy_score,plot_confusion_matrix,auc,confusion_matrix



### Accuracy on the train dataset
rf_y_train_pred = randomModel.predict(x_train)
accuracy_score(y_train, rf_y_train_pred)

### Accuracy on the test dataset
rf_y_test_pred = randomModel.predict(x_test)
accuracy_score(y_test, rf_y_test_pred)

f1_score(y_test, rf_y_test_pred)



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





