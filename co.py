# I import all the libraries I need to solve the problem 

import numpy as np
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score,calinski_harabasz_score

#Load the data and transpose it to have it with the rigth dimensions
data=np.load('.\\data.npy')
data=data.transpose()

#Use scaler so we can have normalized data and reduce bias
scaler = StandardScaler() 
scaler.fit(data)  
data = scaler.transform(data) 

# Iteration process to select the number of Principal components
m=data.shape[1]
n=0
for i in range(0,m):
    pca = PCA(n_components=i+1)
    model = pca.fit(data)
    weights=model.explained_variance_ratio_
    percentage=np.sum(weights)*100
    if percentage>85:
        n=i+1
        break
    
print('Percentage of information captured with',n,'principal components: {:.1f}%'.format(percentage))

# Reduce data to the number of principal components selected before
data_reduced=model.transform(data) 

#Iteration process to select the number of clusters using K-means

k=20 #number of iterations

# Initial vectors for the different scores used
sil=[]
dav=[]
cal=[]
wcss=[]

for i in range(2,k):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init="auto").fit(data_reduced)
    wcss.append(kmeans.inertia_)
    sil.append(silhouette_score(data_reduced,kmeans.labels_))
    dav.append(davies_bouldin_score(data_reduced,kmeans.labels_))
    cal.append(calinski_harabasz_score(data_reduced,kmeans.labels_))

# Plotting the results to analyze and select the best number of clusters 

clusters=np.linspace(2,k,k-2) #vector with the number of clusters in each iteration

plt.figure(1)
plt.plot(clusters,wcss)
plt.title("Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")


plt.figure(2)
plt.plot(clusters,sil)
plt.title("Silhouette score")
plt.xlabel("Number of clusters")

plt.figure(3)
plt.plot(clusters,dav)
plt.title("Davies-Bouldin score")
plt.xlabel("Number of clusters")

plt.figure(4)
plt.plot(clusters,cal)
plt.title("Calinski-Harabasz score")
plt.xlabel("Number of clusters")
plt.show()

# According to the elbow graph the number of clusters should be in the range 3-8
# Regarding the Calinski-Harabasz score 2 or 3 clusters are the best options
# About the Davies-Bouldin the best number of clusters is 3
# Finally, checking the silhouette score 2 or 3 clusteres are the best options
# Therefore the best number of clusters is 3
K=3 

# Fit data with the number of clusters selected and calculate the final scores for each method
kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(data_reduced)
sil_final = silhouette_score(data_reduced,kmeans.labels_)
dav_final = davies_bouldin_score(data_reduced,kmeans.labels_)
cal_final = calinski_harabasz_score(data_reduced,kmeans.labels_)
print('K-means final scores')
print('Silhouette: {:.3f}'.format(sil_final),'(the closer to 1 the better)')
print('Davies Bouldin: {:.3f}'.format(dav_final),'(the closer to 0 the better)')
print('Calinski-Harabasz: {:.3f}'.format(cal_final),'(the higher the better)')

agglo = AgglomerativeClustering(n_clusters=3,linkage='average').fit(data_reduced)
sil_final = silhouette_score(data_reduced,agglo.labels_)
dav_final = davies_bouldin_score(data_reduced,agglo.labels_)
cal_final = calinski_harabasz_score(data_reduced,agglo.labels_)

print('Agglomerative final scores')
print('Silhouette: {:.3f}'.format(sil_final),'(the closer to 1 the better)')
print('Davies Bouldin: {:.3f}'.format(dav_final),'(the closer to 0 the better)')
print('Calinski-Harabasz: {:.3f}'.format(cal_final),'(the higher the better)')

gmm = GaussianMixture(n_components=3,n_init=3,random_state=2).fit_predict(data_reduced)
sil_final = silhouette_score(data_reduced,gmm)
dav_final = davies_bouldin_score(data_reduced,gmm)
cal_final = calinski_harabasz_score(data_reduced,gmm)

print('Gaussian mixture final scores')
print('Silhouette: {:.3f}'.format(sil_final),'(the closer to 1 the better)')
print('Davies Bouldin: {:.3f}'.format(dav_final),'(the closer to 0 the better)')
print('Calinski-Harabasz: {:.3f}'.format(cal_final),'(the higher the better)')

#Comparing the results the agglomerative method performs better than the others