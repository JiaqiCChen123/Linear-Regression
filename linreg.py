import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def normalize(X): # creating standard variables here (u-x)/sigma
# input is a df
    if isinstance(X, pd.DataFrame):
        for c in X.columns:
            if is_numeric_dtype(X[c]):
                u = np.mean(X[c])
                s = np.std(X[c])
                X[c] = (X[c] - u) / s
        return
    for j in range(X.shape[1]):
        u = np.mean(X[:,j])
        s = np.std(X[:,j])
        X[:,j] = (X[:,j] - u) / s
 
def MSE(X,y,B,lmbda):
    transpose_matrix = np.transpose(y - np.dot(X, B))
    matrix = y - np.dot(X, B)
    return np.dot(transpose_matrix, matrix)

def loss_gradient(X, y, B, lmbda):
    matrix = y - np.dot(X, B)
    return -np.dot(np.transpose(X), matrix)

def loss_ridge(X, y, B, lmbda):
    penalty = lmbda * np.dot(np.transpose(B), B)
    return MSE(X, y, B, lmbda) + penalty
    
def loss_gradient_ridge(X, y, B, lmbda):
    return loss_gradient(X, y, B, lmbda) + lmbda*B

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def log_likelihood_gradient(X, y, B,lmbda):
    return (-1)*np.dot(np.transpose(X),(y-sigmoid(np.dot(X,B))))

def log_likelihood(X, y, B,lmbda):
    # list_temp = [y[i]*np.dot(X[i,:],B) - np.log(1 + np.exp(np.dot(X[i,:],B))) for i in range(X.shape[0])]
    # return (-1)*sum(list_temp)
    return -np.sum(y*np.dot(X,B) - np.log(1 + np.exp(np.dot(X,B))))

def minimize(X, y, loss, loss_gradient,
              eta=0.00001, lmbda=0.0,
              max_iter=1000, addB0=True,
              precision=0.00000001):
    "Here are various bits and pieces you might want"
    if X.ndim != 2:
        raise ValueError("X must be n x p for p features")
    n, p = X.shape
    if y.shape != (n, 1):
        raise ValueError(f"y must be n={n} x 1 not {y.shape}")
    if addB0:
        # X.insert(loc=0,value=1,column = 'X0') # add column of 1s to X
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        n, p = X.shape
    B = np.random.random_sample(size=(p, 1)) * 2 - 1  # make between [-1,1)
    prev_B = B  # p*1 vector
    cost = 9e99
    step = 0
    eps = 1e-5 # prevent division by 0

    for i in range(max_iter):
        l_gradiant = loss_gradient(X, y, prev_B, lmbda)
        step += l_gradiant * l_gradiant
        after_B = prev_B - eta*l_gradiant/(eps + np.sqrt(step))
        if np.absolute(loss(X,y,after_B,lmbda)-loss(X,y,prev_B,lmbda)) >= precision:
            prev_B = after_B
        else:
            final_B = prev_B
            break
    final_B = prev_B
    if addB0:
        return final_B 
    else: 
        return np.insert(final_B,0,np.mean(y),axis=0)
    

class LinearRegression:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]  # n
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           MSE,
                           loss_gradient,
                           self.eta,
                           self.lmbda,
                           self.max_iter)

class RidgeRegression:
    def __init__(self,
                eta=0.00001, lmbda=0.0,
                max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]  # n
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                            loss_ridge,
                            loss_gradient_ridge,
                            self.eta,
                            self.lmbda,
                            self.max_iter,addB0=False)

class LogisticRegression:
    def __init__(self,
                 eta=0.00001, lmbda=0.0,
                 max_iter=1000):
        self.eta = eta
        self.lmbda = lmbda
        self.max_iter = max_iter

    def predict(self, X):
        n = X.shape[0]  # n
        B0 = np.ones(shape=(n, 1))
        X = np.hstack([B0, X])
        return np.where(sigmoid(np.dot(X,self.B))>0.5,1,0)
        # return np.dot(X, self.B)

    def fit(self, X, y):
        self.B = minimize(X, y,
                           log_likelihood,
                           log_likelihood_gradient,
                           self.eta,
                           self.lmbda,
                           self.max_iter,addB0=True)
        
