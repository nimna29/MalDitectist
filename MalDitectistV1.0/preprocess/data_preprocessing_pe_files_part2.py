####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Data Pre-processing PE Files - Part 2####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_csv('../combined_pe_files_dataset.csv')
dataset.shape
dataset.describe()

# Find Outliers loop for plots
## Remove some columns
dataset = dataset.drop(['file_name', 'machine_type', 'timestamp', 'pointer_to_symbol_table', 'number_of_symbols'], axis=1)

# for column in dataset.columns:
#     column_data = dataset[column]
    
#     plt.hist(column_data)
#     plt.title(column)
#     plt.show()

#     lowerLimit = column_data.quantile(0.02)
#     lowerLimit
#     dataset[column_data < lowerLimit]
    
#     upperLimit = column_data.quantile(0.98)
#     upperLimit
#     upper_values = dataset[column_data > upperLimit]
    
#     ## Value Counter
#     low_value_count = np.sum(column_data < upperLimit)
#     print(f'Lower Value count of {column}: {low_value_count}')
    
#     up_value_count = np.sum(column_data > upperLimit)
#     print(f'Upper Value count of {column}: {up_value_count}')
#     print('')


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
# low_value_count = np.sum(column_name <= upperLimit)
# print(f'Upper Value count: {low_value_count}')

# up_value_count = np.sum(column_name >= upperLimit)
# print(f'Upper Value count: {up_value_count}')


## --------- ##


# Create dependent & independent variable vectors
x = dataset.iloc[:, :-1].values #independent - other features (removed classification_list)
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


# # Create a new DataFrame with preprocessed data
# preprocessed_df = pd.DataFrame(data=x, columns=dataset.columns[1:-1])

# # Add the classification column to the preprocessed DataFrame
# preprocessed_df['classification_list'] = y

# # Save preprocessed dataset as CSV file
# # preprocessed_df.to_csv('/home/nimna/Documents/MyProjects/DatasetMalDitectist/preprocessed_dataset.csv', index=False)
# # print("Dataset Creation Successfully Completed!!!")

# ### Completed Data Preprocessing - App Version 1.0.0 ###
