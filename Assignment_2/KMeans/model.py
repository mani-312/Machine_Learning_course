import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        (n,c) = X.shape
        self.cluster_centers = X[np.random.choice(range(0, n), size=self.num_clusters, replace=False)]
        #self.cluster_centers = np.random.uniform(low=-2, high=2, size=(self.num_clusters,c))
        assignment = np.zeros((n,1),dtype = int)
        pass

        for iteration in range(max_iter):
            print("     Iteration == ",iteration)
            # Assign each sample to the closest prototype
            for i in range(n):
                pixel = X[i]
                # Find the closest cluster to pixel
                diff = self.cluster_centers - pixel.reshape(1,-1)
                
                cluster_distances = np.diag(diff @ np.transpose(diff)).reshape(-1,1) # k*1
                assignment[i] = np.argmin(cluster_distances)
            #pass
            
            self.cluster_centers = np.zeros((self.num_clusters,c))
            count = np.zeros((self.num_clusters,1),dtype = int)
            for i in range(n):
                pixel = X[i]
                cluster = assignment[i]
                #print(cluster)
                count[cluster]+=1
                self.cluster_centers[cluster]+=pixel
            count[count==0]+=1 # Handling the case divide by 0
            self.cluster_centers = self.cluster_centers/count
            #print(self.cluster_centers)
            # Update prototypes

            #pass

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        raise NotImplementedError
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        assignment = np.zeros(X.shape)
        (n,c) = X.shape
        for i in range(n):
            pixel = X[i]
            # Find the closest cluster to pixel
            diff = self.cluster_centers - pixel.reshape(1,-1)
                
            cluster_distances = np.diag(diff @ np.transpose(diff)).reshape(-1,1) # k*1
            assignment[i] = self.cluster_centers[np.argmin(cluster_distances)]
        return assignment
        raise NotImplementedError