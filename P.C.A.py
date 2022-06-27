"""
Created on Tue May 20 16:52:47 2022

@author: lalith kumar
"""
#Performing Principal component analysis and performing clustering using first 3 principal component scores.

# import datasets
import pandas as pd
df=pd.read_csv('E:\data science\ASSIGNMENTS\ASSIGNMENTS\P.C.A\wine.csv')
df.shape
list(df)
df.head()
df.describe()
df.info()

# droping the 'type' variable.
x1=df.drop(('Type'),axis=1)
list(x1)
x1.shape
x1.head()

# checking correlation& covariance.
df.corr()
df.cov()
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8));sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(12,8));sns.heatmap(df.cov(),annot=True)

# standardlization.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler() 
x1_scaled=sc.fit_transform(x1)

# load decomposition to do PCA analysis with sklearn
from sklearn.decomposition import PCA
PCA()
pca = PCA()

# pca variance ratio.
pc = pca.fit_transform(x1_scaled)
var=pca.explained_variance_ratio_
var
'''
array([0.36198848, 0.1920749 , 0.11123631, 0.0706903 , 0.06563294,
       0.04935823, 0.04238679, 0.02680749, 0.02222153, 0.01930019,
       0.01736836, 0.01298233, 0.00795215])'''
# checking the sum of ratio.
sum(pca.explained_variance_ratio_)

pc.shape
pd.DataFrame(pc).head()
type(pc)

pca = PCA(n_components = 3)
pca_values = pca.fit_transform(x1)


pc_df = pd.DataFrame(data = pc , columns = ['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P0C9','P0C10','P0C11','P0C12','P0C13'])
pc_df.head()
pc_df.shape
type(pc_df)
# describe
pc_df["P0C1"].describe()
pc_df["P0C2"].describe()
pc_df["P0C3"].describe()

pc = pca.fit_transform(x1_scaled)
var=pca.explained_variance_ratio_
var

import matplotlib.pyplot as plt
PC = range(1, pca.n_components_+1)
plt.bar(PC, var, color='BLACK')
plt.xlabel('Principal Components')
plt.ylabel('Variance %')
plt.xticks(PC)

# hence,we need to work on only three components.
pca= PCA(n_components=3)
Pc=pca.fit_transform(x1_scaled)

# checking the ratio of 3 component.
var_exp=pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
# 0.6652996889318524

Pc.shape
pd.DataFrame(Pc).head()
type(Pc)

#visualizing
#plt pca explained ratio
plt.plot(var,color="GREEN")

#plot pca1 and pca2
PCA_components = pd.DataFrame(Pc)
plt.scatter(PCA_components[0], PCA_components[1], alpha=.3, color='blue')
plt.xlabel('P0C1')
plt.ylabel('POC2')
plt.show()

# kmeans.
# elbow curve.
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(PCA_components.iloc[:,:3])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow curve')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


new_df= pd.DataFrame(Pc)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
labels=kmeans.labels_

#scatter plot
plt.scatter(PCA_components[0], PCA_components[1], c=labels)
plt.show()

model_df=pd.Series(labels)
x1["cluster_kmean"]=model_df
x1
x1.groupby(x1.cluster_kmean).mean()
#--------------------------------------------------------

# heirarchial Clustering.
# import hierarchy. 
import scipy.cluster.hierarchy as shc

# ploting Dendogram.

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))  
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
dend = shc.dendrogram(shc.linkage(Pc, method='complete')) 

# creating a group using clusters.

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(Pc)

# ploting. 
plt.figure(figsize=(10, 7))  
plt.scatter(Pc[:,0], Pc[:,1], c=cluster.labels_, cmap='rainbow')  

Y_clust_H = pd.DataFrame(Y)
Y_clust_H[0].value_counts()

x1['clust_H']=Y_clust_H
x1
x1.groupby(x1.clust_H).mean()

datanew=x1
datanew.head()

#====================================================================













