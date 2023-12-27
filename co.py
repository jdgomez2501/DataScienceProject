# I import all the libraries I need to solve the problem 

import numpy as np
from sklearn.preprocessing import StandardScaler  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
    if percentage>95:
        n=i+1
        break
    

print('Percentage of information captured with',n,'principal components: {:.1f}%'.format(percentage))
#89.5 with 7, 94 with 8 y 97 with 9

# Reduce data to the number of principal components selected before
data_reduced=model.transform(data) 


#Iteration process to select the number of clusters

k=50 #number of iterations

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
plt.title("WCSS")
plt.xlabel("Number of clusters")

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
plt.title("Calinski score")
plt.xlabel("Number of clusters")
plt.show()

K=4 #According to the plots this is the best number of clusters

# Fit data with the number of clusters selected and calculate the final scores
kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(data_reduced)
sil_final = silhouette_score(data_reduced,kmeans.labels_)
dav_final = davies_bouldin_score(data_reduced,kmeans.labels_)
cal_final = calinski_harabasz_score(data_reduced,kmeans.labels_)
wcss_final = kmeans.inertia_

print('Silhouette final score: {:.1f}'.format(sil_final),'(the closer to 1 the better)')
print('Davies Bouldin final score: {:.1f}'.format(dav_final),'(the closer to 0 the better)')
print('Calinski final score: {:.1f}'.format(cal_final),'(the higher the better)')
print('Final wcss: {:.1f}'.format(wcss_final),'(the less the better)')
