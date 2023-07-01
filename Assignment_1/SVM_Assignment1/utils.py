import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data

    X_train_min = np.min(X_train, axis = 0)
    X_train_max = np.max(X_train, axis = 0)
    den = X_train_max - X_train_min
    den[den==0] = 1
    X_train_norm = 2*((X_train-X_train_min)/den)-1

    X_test_min = np.min(X_test, axis = 0)
    X_test_max = np.max(X_test, axis = 0)
    den = X_test_max - X_test_min
    den[den==0] = 1
    X_test_norm = 2*((X_test-X_test_min)/den)-1

    return (X_train_norm,X_test_norm)
    
    raise NotImplementedError


def plot_metrics(metrics) -> None:
    # plot and save the results
    K = []
    precision = []
    recall = []
    f1_score = []
    accuracy = []
    for metric in metrics:
        K.append(metric[0])
        accuracy.append(metric[1])
        precision.append(metric[2])
        recall.append(metric[3])
        f1_score.append(metric[4])

    plt.figure(0)
    plt.plot(K,accuracy,'-o')
    plt.plot(K,precision,'-o')
    plt.plot(K,recall,'-o')
    plt.plot(K,f1_score,'-o')
    plt.xlabel('Number of Principal Componnets')
    plt.ylabel('Score')
    plt.legend(['Accuracy','Precision','Recall','F1_Score'])
    #plt.title('Learning_rate = '+str(learning_rate)+', Num_iters = '+str(num_iters)+', C = '+str(C))
    plt.savefig('plot.jpg')
    plt.clf()

    return None
    raise NotImplementedError