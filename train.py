import csv
import numpy as np

def import_data():
    X = np.genfromtxt("train_X_lr.csv",delimiter=',',dtype = np.float64,skip_header = 1)
    Y = np.genfromtxt("train_Y_lr.csv",delimiter=',',dtype = np.float64)
    return X, Y

def compute_gradient_of_cost_functoin(X,Y,W):
    Y_pred = np.dot(X,W)
    difference = Y_pred - Y
    dW = (1/len(X))*(np.dot(difference.T,X))
    dW = dW.T
    return dW

def compute_cost(X,Y,W):
    Y_pred = np.dot(X,W)
    mse = np.sum(np.square(Y_pred - Y))
    cost_value = mse/(2*len(X))
    return cost_value

def optimize_weights_using_gradient_descent(X,Y,W,num_iter,alpha):

    prev_iter_cost = 0
    iter_cnt = 0
    while True:
        iter_cnt += 1
        dW = compute_gradient_of_cost_functoin(X,Y,W)
        W = W - (alpha * dW)
        cost = compute_cost(X,Y,W)

        #if iter_cnt%500000 == 0:
        #    print(iter_cnt,cost,abs(prev_iter_cost - cost))
            
        if abs(prev_iter_cost - cost) < 0.000001 or cost < 100:
        #    print(iter_cnt,cost)
            break
        prev_iter_cost = cost

    return W

def train_model(X,Y):
    X = np.insert(X,0,1,axis = 1)
    Y = Y.reshape(len(X),1)
    W = np.zeros((X.shape[1],1))
    W = optimize_weights_using_gradient_descent(X,Y,W,20000000,0.0002)

    return W

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    X,Y = import_data()
    weights = train_model(X,Y)
    save_model(weights,"WEIGHTS_FILE.csv")
