####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Model Building 2 - Unsupervised Learning v 1.0.0 ####
# K-Means Clustering

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the preprocessed dataset
dataset = pd.read_csv('../preprocessed_dataset.csv')
print("Original dataset shape:", dataset.shape)

# Remove unnecessary columns
# Load the preprocessed dataset and drop unnecessary columns
dataset = dataset.drop(['total_tool_detection', 'tool_detection_positives'], axis=1)
print("Dataset shape after removing unnecessary columns:", dataset.shape)

# Split the dataset into features (X) and labels (y)
X = dataset.drop(['classification_list'], axis=1)
y_true = dataset['classification_list']

# Train the k-means model with k=2
kmeans = KMeans(n_clusters=2, n_init=25, random_state=42) # default is n_init = 10
kmeans.fit(X)

# Reduce the dimensionality of the dataset using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
plt.title('K-means Clustering Results')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.show()

# Evaluate the performance of the model
y_pred = kmeans.predict(X)
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Calculate and print the accuracy score of the model
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate f1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_true, y_pred, average='binary')
print("F1 Score: ", f1)

# Print detailed results and analysis
print(f"\nNumber of samples in each cluster: {pd.Series(kmeans.labels_).value_counts()}")
print(f"\nCentroids of the clusters:\n{pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)}")
print(f"\nExplained variance ratio by the first two principal components: {pca.explained_variance_ratio_}")

# Plot the explained variance ratio by the principal components
plt.plot(pca.explained_variance_ratio_)
plt.title('Explained Variance Ratio')
plt.xlabel('PCA Component')
plt.ylabel('Variance Ratio')
plt.show()