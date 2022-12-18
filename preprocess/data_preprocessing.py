####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Data Pre-processing ####

import numpy as np
import pandas as pd

# Load dataset
dataset = pd.read_csv('../dataset1.csv')
print(dataset.describe())


# Create dependent & independent variable vectors
x = dataset.iloc[:,4:-4].values #independent
y = dataset.iloc[:,-3].values #dependent


# Handle missing data
# Count the num of missing values in each column - output = 0
# print(dataset.isnull().sum()) 


# Data Encoding
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



