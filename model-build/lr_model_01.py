####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Logistic Regression - Default: lbfgs Model ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset - Used preprocessed dataset that created using data_preprocessing_pe_files.py
dataset = pd.read_csv('../preprocessed_dataset.csv')

# Create dependent & independent variable vectors
x = dataset.iloc[:, :-3].values #independent - pe file features
y = dataset.iloc[:,-1].values #dependent - classification_list (Blacklist = 1 / Whitelist = 0)

# x = dataset.iloc[:, :-1].values  (removed only classification_list )
# x = dataset.iloc[:, :-3].values (removed total_tool_detection, tool_detection_positives, classification_list)

# Split the dataset for training & testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=42)

print("Train Shape:",x_train.shape, "\nTest Shape:", x_test.shape)


## Logistic Regression - Default: lbfgs
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0, max_iter=1000, penalty='l2')
logModel = lr.fit(x_train, y_train) 


### Model Evaluation
from sklearn.metrics import f1_score,accuracy_score,plot_confusion_matrix,confusion_matrix,r2_score,precision_score

### Accuracy on the train dataset
lr_y_train_pred = logModel.predict(x_train)
trainAS = round(accuracy_score(y_train, lr_y_train_pred)*100, 4)

### Accuracy on the test dataset - validation accuracy
lr_y_test_pred = logModel.predict(x_test)
testAS = accuracy_score(y_test, lr_y_test_pred)
val_loss = round((1 - testAS)*100, 4)
val_accurcy = round(testAS*100, 4)

# Calculate precision
precision = round(precision_score(y_test, lr_y_test_pred)*100, 4)

# Calculate F1 Score
f1 = round(f1_score(y_test, lr_y_test_pred)*100, 4)

# Calculate R2 score
r2 = round(r2_score(y_test, lr_y_test_pred)*100, 4)


print("\nAccuracy on the train dataset:", trainAS)
print("Validation Accuracy:", val_accurcy)
print("Validation Loss:", val_loss)
print("Precision:", precision)
print("F1 Score:", f1)
print("R2 Score:", r2)



## Confusion matrix - Logistic Regression - Default: lbfgs
### Confusion matrix without Normalization
lr_conf_matrix = confusion_matrix(y_test, lr_y_test_pred)

### Confusion matrix with Normalization
lr_conf_matrix_norm = confusion_matrix(y_test, lr_y_test_pred, normalize='true')


# Calculate TP, TN, FP, FN
TP = round(lr_conf_matrix_norm[1, 1]*100, 4)
TN = round(lr_conf_matrix_norm[0, 0]*100, 4)
FP = round(lr_conf_matrix_norm[0, 1]*100, 4)
FN = round(lr_conf_matrix_norm[1, 0]*100, 4)

print("\nTrue Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)


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