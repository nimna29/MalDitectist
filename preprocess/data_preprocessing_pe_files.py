####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Data Pre-processing PE Files ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('../combined_pe_files_dataset.csv')
dataset.shape
dataset.describe()

# # Find Outliers loop for plots
# for column in dataset.columns:
#     column_data = dataset[column]
    
#     plt.hist(column_data)
#     plt.title(column)
#     plt.show()


# # Find Outliers
# column_name = dataset['machine_type']

# plt.hist(column_name)
# plt.title("machine_type")
# plt.show()

# ## Quantile
# lowerLimit = column_name.quantile(0.05)
# lowerLimit
# dataset[column_name < lowerLimit]

# upperLimit = column_name.quantile(0.95)
# upperLimit
# upper_values = dataset[column_name > upperLimit]

# ## Value Counter
# value_count = np.sum(column_name == 43620)
# print(f'Value count: {value_count}')


## --------- ##


# Create dependent & independent variable vectors
x = dataset.iloc[:,1:-1].values #independent - other features (removed file_name & classification_list)
y = dataset.iloc[:,-1].values #dependent - classification_list

# Handle missing data
# Count the num of missing values in each column - output = 0
# print(dataset.isnull().sum()) 

# Data Encoding
## Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(y)
y = np.where(y == 0, 1, 0) # Blacklist = 1 / Whitelist = 0

# Feature Scaling - Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)


# Create a new DataFrame with preprocessed data
preprocessed_df = pd.DataFrame(data=x, columns=dataset.columns[1:-1])

# Add the classification column to the preprocessed DataFrame
preprocessed_df['classification_list'] = y

# Save preprocessed dataset as CSV file
preprocessed_df.to_csv('/home/nimna/Documents/MyProjects/DatasetMalDitectist/preprocessed_dataset.csv', index=False)
print("Dataset Creation Successfully Completed!!!")
