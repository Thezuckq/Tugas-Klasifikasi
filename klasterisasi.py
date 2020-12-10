import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

col_list = ['urlDrugName', 'rating', 'effectiveness', 'sideEffects', 'condition']
drugReview = pd.read_csv('drugReview.csv', usecols = col_list)
pd.set_option('display.max_columns',None)

print(drugReview.head())

print(drugReview.info()) #Menampilkan Berbagai Informasi dataset

print("Drug Review data set dimensions : {}".format(drugReview.shape))

# print(drugReview.isna().sum()) #Melihat jumlah Missing Value

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

# columnDrugName = drugReview.iloc[:,0]
# plt.subplot(columnDrugName,bins=10,color='blue')
# plt.show()

# sb.scatterplot(x="urlDrugName",y="rating",data=drugReview,s=100,color="blue",alpha=0.5)
# plt.show()

