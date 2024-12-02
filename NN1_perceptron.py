#task - Build a Perceptron from first principles using numpy, matplotlib
# use it as a classifer 
from sklearn.datasets import make_classification
import numpy as np 
import matplotlib.pyplot as plt 

# 1. Creating the data-set 

D,y = make_classification(n_samples = 100,n_features = 2, n_informative=1,
    n_redundant=0,n_classes=2, n_clusters_per_class=1, random_state=41,hypercube=False,class_sep=10)

# D is the coordinates (x1,x2)
# y is the class of each point 

# plt.figure(figsize=(10,10))
# plt.scatter(D[:,0], D[:,1],c=y,cmap='winter',s=100)
# plt.show()

# 2. Creating the fundamental structure of a perceptron 

def step(x):
    if x>=0:
        return 1
    else :
        return 0 

def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

def perceptron(D,y):
    Data = np.insert(D,0,1,axis =1)
    weights = np.random.rand(Data.shape[1]) #no. of dimensions 
    learn_rate = 0.1
    
    output = np.zeros(Data.shape[0])
    convergence = 1 
    
    while convergence > 1e-2:
        RSE = 0 #residual sum of errors
        for i in range(Data.shape[0]):
            fi = np.dot(weights.T , Data[i])
            output[i] = sigmoid(fi) #step should be used 
            error = y[i]-output[i] 
            
            for j in range(Data.shape[1]):
                weights[j] = weights[j] + learn_rate*(error)*Data[i][j]
            
            RSE += abs(error)
        convergence = RSE/(Data.shape[1])
        
    return weights[0] , weights[1:]

intercept, coeffs = perceptron(D,y)


# for 2D inputs : y = m * x + c , hence convert w0 + w1x1 + w2x2 = 0 to the standard equation 
# std eqn : x2 = (-w1/w2)x1 + (-w0)/w2

slope = -coeffs[0]/coeffs[1]
intercept = -intercept/coeffs[1]


# Printing the slope and intercept of the line 
print(f"Slope : {slope} , Intercept : {intercept}")

#Plotting the decision boundary 
x_vals = np.linspace(D[:, 0].min(), D[:, 0].max(), 100)
y_vals = slope * x_vals + intercept
plt.figure(figsize=(10, 10))
plt.scatter(D[:, 0], D[:, 1], c=y, cmap='winter', s=100)
plt.plot(x_vals, y_vals, color='red', label="Decision Boundary")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.show()
