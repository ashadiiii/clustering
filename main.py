import numpy as np
import matplotlib.pyplot as plt

class kMeansClustering:
    
    def __init__(self,k=3):
        self.k=k
        self.centroids=None
    
    @staticmethod
    def euclidean_distance(datapoint,centroids):
      """
      This method calculates the Euclidean distance between a single 'datapoint' and an array of cetroids.
      parameters : 
      - datapoint: This is a single data point(typically represented as an array or list) for which we want to calculate distances to each centroid. 
      - centroids: An array if cebtroids. each centroid is represented as a point in the same dimentional space as 'datapoint'.

      function implemenation:
      - Squared Differences: (centroids - datapoint) ** 2 calculates the squared differences between each centroid and the datapoint. This is a vectorized operation in NumPy, meaning it efficiently computes the squared differences element-wise.
      - Summation: np.sum(..., axis=1) sums up the squared differences along axis 1, which corresponds to summing across the dimensions of the datapoint.
      - Square Root: np.sqrt(...) takes the square root of the summed squared differences, resulting in the Euclidean distance.

      return value:
      - The method returns an array containing the Euclidean distances between the 'data point' and each centroid in the 'centroids' array.
      """
      return np.sqrt(np.sum((centroids-datapoint)**2,axis=1))

    def fit(self,x,max_iterations=200):
      #initialize the 'self.centroids' randomly within the range of the data points (x)
      self.centroids = np.random.uniform(np.amin(x,axis=0),np.amax(x,axis=0),size=(self.k,x.shape[1]))

      for _ in range(max_iterations):
          
        y= [] #to store cluster assignments for each datapoint
        for data_point in x:
            distances= kMeansClustering.euclidean_distance(data_point,self.centroids) #computes distances to all centroids using 'euclidean_distance' method.
            cluster_num = np.argmin(distances) #finds the index with the smallest distance and appends it to the the list 'y'.
            y.append(cluster_num)

        y = np.array(y)
        cluster_indices = [] # a list where each element corresponds to indices of data points assigned to each cluster.
        for i in range(self.k):
           cluster_indices.append(np.argwhere(y==i))
        cluster_centers = [] # a list that will store updated centroid positions
        for i,indices in enumerate(cluster_indices):
           if len(indices)==0: 
              cluster_centers.append(self.centroids[i]) #if there are no points the centroid remains unchanged
           else:
              cluster_centers.append(np.mean(x[indices],axis=0)[0]) #realigns the centroid to be in the mean of data points in each cluster
        if np.max(self.centroids-np.array(cluster_centers))<0.0001: #Checks for convergence by comparing the maximum distance between old and new centroids. If centroids have converged, breaks out of the loop.
           break
        else:
           self.centroids = np.array(cluster_centers) #reassigns the centroid points to the ones calculated above(stored in the cluster_centers list)
      return y
    
random_points = np.random.randint(0,100,(100,2))

kmeans = kMeansClustering(k=3)
labels = kmeans.fit(random_points)

plt.scatter(random_points[:,0],random_points[:,1],c=labels)
plt.scatter(kmeans.centroids[:,0],kmeans.centroids[:,1],c=range(len(kmeans.centroids)),marker="*",s=200)


plt.show()
