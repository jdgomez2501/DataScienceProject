# I import all the libraries I need to solve the problem 

import numpy as np
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score

#Load the data and transpose it to have it with the rigth dimensions
data=np.load('.\\Project\\Project\\data.npy')
data=data.transpose()

#Use scaler so we can have normalized data and reduce bias
scaler = StandardScaler() 
scaler.fit(data)  
data = scaler.transform(data) 


m=data.shape[1]
n=0
for i in range(0,m):
    pca = PCA(n_components=i+1)
    model = pca.fit(data)
    weights=model.explained_variance_ratio_
    percentage=np.sum(weights)*100
    if percentage>95:
        n=i+1
        break
print(percentage,n)#89.5 con 7, 94 con 8 y 97 con 9
data_reduced=model.transform(data)
k=50
sil=[] #the closer to 1 the better
dav=[] # the closer to 0 the better
cal=[] #the higher the better
wcss=[] #the less the better
for i in range(2,k):   
    kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(data_reduced)
    wcss.append(kmeans.inertia_)
    sil.append(silhouette_score(data_reduced,kmeans.labels_))
    dav.append(davies_bouldin_score(data_reduced,kmeans.labels_))
    cal.append(calinski_harabasz_score(data_reduced,kmeans.labels_))
clusters=np.linspace(2,k,k-2)
plt.figure(1)
plt.plot(clusters,wcss)
plt.figure(2)
plt.plot(clusters,sil)
plt.figure(3)
plt.plot(clusters,dav)
plt.figure(4)
plt.plot(clusters,cal)
plt.show()

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(data_reduced)
sil_final = silhouette_score(data_reduced,kmeans.labels_)
dav_final = davies_bouldin_score(data_reduced,kmeans.labels_)
cal_final = calinski_harabasz_score(data_reduced,kmeans.labels_)
wcss_final = kmeans.inertia_
print(kmeans.cluster_centers_)
print(sil_final,dav_final,cal_final,wcss_final)
""" print(kmeans.labels_)
kmeans.predict([])
print(kmeans.cluster_centers_)
print(kmeans.inertia_) """