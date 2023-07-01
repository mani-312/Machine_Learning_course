import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        S = np.cov(X, rowvar=False)
        eigenValues, eigenVectors = np.linalg.eig(S)
        sum = np.sum(eigenValues)
        idx = eigenValues.argsort()[::-1] # eigenValues in decreasing order
        eigenValues = eigenValues[idx]
        var = np.sum(eigenValues[:self.n_components])/sum
        eigenVectors = eigenVectors[:,idx]

        # Largest n_components eigen values
        self.components = eigenVectors[:,0:self.n_components]

        print("Variance Explained == ",var)
        return None
        raise NotImplementedError
    
    def transform(self, X) -> np.ndarray:
        # transform the data

        return X @ self.components
        raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        (N,D) = X.shape
        self.w = np.zeros((D,1))
        self.b = 0.0
        pass

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            rand_idx = np.random.choice(X.shape[0])
            xn = X[rand_idx, :].reshape(-1,1)
            yn = y[rand_idx]

            hinge_loss = max(0,1-yn*(np.dot(np.transpose(xn),self.w)+self.b))
            delW = self.w
            delb = 0
            if hinge_loss > 0:
                delW = self.w + C*(-yn*xn)
                delb = C*(-yn)

            self.w = self.w - learning_rate*delW
            self.b = self.b - learning_rate*delb 
        return None
        raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        pred = X@self.w + self.b
        pred[pred>=0] = 1
        pred[pred<0] = -1
        
        return pred
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        tp = np.sum((y==1) & (y_pred==1))
        predp = np.sum(y_pred==1)
        eps = 1e-7
        precision_label = tp/(predp+eps)
        
        return precision_label
    
    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        tp = np.sum((y==1) & (y_pred==1))
        predp = np.sum(y==1)
        eps = 1e-7
        recall_label = tp/(predp+eps)
        
        return recall_label
    
    def f1_score(self, X, y) -> float:
        precision = self.precision_score(X,y)
        recall = self.recall_score(X,y)
        eps = 1e-7
        return 2 * (precision * recall) / (precision + recall+eps)

class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class

        for label in range(self.num_classes):
            X_train = X
            y_train = y.copy()
            y_train[y==label] = 1
            y_train[y!=label] = -1
            self.models[label].fit(X_train,y_train,kwargs['learning_rate'],kwargs['num_iters'],kwargs['C'])
        return None
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        result = np.zeros((X.shape[0],self.num_classes))
        for label in range(self.num_classes):
             result[:,label] = np.squeeze(X@self.models[label].w + self.models[label].b)

        return np.argmax(result,axis = 1)
        # pass the data through all the 10 SVM models and return the class with the highest score
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        return np.mean(self.predict(X) == y)
    
    def precision_score(self, X, y) -> float:
        y_pred = self.predict(X)
        #print(y_pred.shape,y.shape)
        precision = 0.0
        eps = 1e-7
        for label in range(self.num_classes):
            tp = np.sum((y==label) & (y_pred==label))
            predp = np.sum(y_pred==label)
            precision_label = tp/(predp+eps)

            precision = precision+precision_label

        return precision/self.num_classes
        raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        y_pred = self.predict(X)
        recall = 0.0
        eps = 1e-7
        for label in range(self.num_classes):
            tp = np.sum((y==label) & (y_pred==label))
            predp = np.sum(y==label)
            recall_label = tp/(predp+eps)

            recall = recall + recall_label

        return recall/self.num_classes
        raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        precision = self.precision_score(X,y)
        recall = self.recall_score(X,y)
        eps = 1e-7
        return 2 * (precision * recall) / (precision + recall+eps)
        raise NotImplementedError
