import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb

from sklearn import preprocessing as pre
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


col_list = ['urlDrugName', 'rating', 'effectiveness', 'sideEffects', 'condition']
drugReview = pd.read_csv('drugReview.csv', usecols = col_list)
pd.set_option('display.max_columns',None)

print(drugReview.head())

print(drugReview.info()) #Menampilkan Berbagai Informasi dataset

print("\nDrug Review data set dimensions : {}".format(drugReview.shape, '\n'))

print("Missing values: \n", drugReview.isna().sum()) #Melihat jumlah Missing Value

drugReview = drugReview.dropna(axis=0)

# print(drugReview.isna().sum()) #Melihat jumlah Missing Value

print("Drug Review data set dimensions : {}".format(drugReview.shape))

print("Drug Name Unique Counts: {}".format(drugReview.urlDrugName.nunique()))
print(drugReview.rating.nunique())
print(drugReview.effectiveness.nunique())
print(drugReview.sideEffects.nunique())
print(drugReview.condition.nunique())

df = drugReview['urlDrugName'].value_counts()
print(df.head(20))

## Encode Data
encode = pre.LabelEncoder()
drugReview['effectiveness'] = encode.fit_transform(drugReview['effectiveness'])
drugReview['sideEffects'] = encode.fit_transform(drugReview['sideEffects'])
drugReview['condition'] = encode.fit_transform(drugReview['condition'])

# Menampilkan nilai korelasi keseluruhan dengan heatmap
corr = drugReview.corr()
ax = sb.heatmap(
    corr, annot = True, fmt='.2f',
    vmin=-1, vmax=1, center=0,
    cmap=sb.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

## Menentukan Features yang akan digunakan dan Menampilkan nilai korelasinya
features = ['rating', 'effectiveness']
features = drugReview[features]

corr = features.corr()
ax = sb.heatmap(
    corr, annot = True, fmt='.2f',
    vmin=-1, vmax=1, center=0,
    cmap=sb.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
plt.show()

# columnDrugName = drugReview.iloc[:,0]
# plt.subplot(columnDrugName,bins=10,color='blue')
# plt.show()

# sb.scatterplot(x="urlDrugName",y="rating",data=drugReview,s=100,color="blue",alpha=0.5)
# plt.show()

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(features)

# Visualising the clusters
plt.scatter(features[y_kmeans == 0, 0], features[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(features[y_kmeans == 1, 0], features[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(features[y_kmeans == 2, 0], features[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(features[y_kmeans == 3, 0], features[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

