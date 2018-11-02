# Import modules
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sklearn.datasets
import statistics
import math
from sklearn.metrics import accuracy_score


# Make plots look pretty
matplotlib.style.use('ggplot')

# Generate dataset with `sklearn.datasets`
np.random.seed(0)
x, y = sklearn.datasets.make_gaussian_quantiles(n_samples=600, n_classes=5)
train_x = x[:500]
train_y = y[:500]
test_x = x[500:]
test_y = y[500:]


def plot_decision_boundary(clf, x, y):
    padding = 0.15
    resolution = 0.01
    #print("x=", x[:5])
    #print("y=", y[:5])

    # Feature range
    x0_min, x0_max = x[:,0].min(), x[:,0].max()
    x1_min, x1_max = x[:,1].min(), x[:,1].max()
    x0_range = x0_max - x0_min
    x1_range = x1_max - x1_min

    # Add padding
    x0_min -= x0_range * padding
    x0_max += x0_range * padding
    x1_min -= x1_range * padding
    x1_max += x1_range * padding

    # Create a meshgrid of points with the above ranges
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, resolution),
                           np.arange(x1_min, x1_max, resolution))
    #print("xx0=", xx0[:5])
    #print("xx1=", xx1[:5])

    # Use `clf` to predict each point of the meshgrid
    # `ravel()` turns a 2D array into a vector
    # `c_` concatenates vectors

    yy = clf.predict(np.c_[xx0.ravel(), xx1.ravel()])
    yy = np.array(yy)
    # Reshape the 1D predictions back to a 2D meshgrid
    yy = yy.reshape(xx0.shape)

    # Plot the contours on the grid
    plt.figure(figsize=(8,6))
    cs = plt.contourf(xx0, xx1, yy, cmap=plt.cm.Spectral)

    # Plot the original data
    plt.scatter(x[:,0], x[:,1], s=35, c=y, cmap=plt.cm.Spectral)

    plt.show()

class MyGaussianNB(object):
    # implement 

    def __init__(self):
        pass

    def split_dataset(self, x):
        # Seperate x by their corresponding label (y). There are 5 classes in our dataset.
        # Therefore, the output of this function is a dictionary with 5 keys:
        # {
        #   0: [[0.1, 0.5], [0.7, 0.05], ...],
        #   1: [a point with label 1, another point with label 1, ...],
        #   ....
        # }
        #pass # TODO: remove pass and implement this function
        separated = {}
        for i in range(len(x)):
            vector = x[i]
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        #print(separated[4])
        return separated





    def summarize(self, dataset):
        summaries = [(statistics.mean(attribute), statistics.stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries

    def compute_gaussian_params(self,data):
        # Input: points with the same label
        # Output: Gaussian parameters (mean & variance) of each feature
        # The output for our 2D dataset is a list of 2 tuples:
        # [(f1 mean, f1 var), (f2 mean, f2 var)]
        #pass # TODO: remove pass and implement this function
        summaries = [(statistics.mean(attribute), statistics.stdev(attribute)) for attribute in zip(*data)]
        del summaries[-1]
        #summaries = {}
        #for classValue, instances in data.items():
        #    summaries[classValue] = self.summarize(instances)
        #print(summaries)
        return summaries

    def fit(self, x, y):
        x = x.tolist()
        data = []
        for ab in zip(x, y):
            temp =[]
            for a in ab:
                if isinstance(a,list):
                    temp = a
                else:
                    temp.append(a)
            #print(temp)
            data.append(temp)
        seperated_by_class = self.split_dataset(data)
        self.gaussian_params = {}
        for label, data in seperated_by_class.items():
            self.gaussian_params[label] = self.compute_gaussian_params(data)
        # After this function, `self.gaussian_params` looks like
        # {
        # ------------------------------------------
        #   class: [fea1(mean, var), fea(mean, var)]
        # ------------------------------------------
        #   0    : [(0.5, 0.28), (0.6, 0.08)],
        #   1    : [(0.3, 0.04), (0.7, 0.14)],
        #   ...
        # }
        print(self.gaussian_params)

    def calculateProbability(self,x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean,2)/(2*math.pow(stdev,2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


    def calculate_prob(self, x):
        # Use self.gaussian_params to compute a probability
        #pass # TODO: remove pass and implement this function
        probabilities = {}
        for classValue, classSummaries in self.gaussian_params.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                t = x[i]
                probabilities[classValue] *= self.calculateProbability(t, mean, stdev)
        pro_list = [0,0,0,0,0]
        for key, value in probabilities.items():
            pro_list[key] = value
        #return probabilities
        #print(probabilities)
        #print("pro_list=", pro_list)
        return pro_list


    def predict(self, x):
        # Predict x, where x looks like [[0.12, 1.5], [0.56, 3.2], ...]
        y_pred = []
        #print("x=", x[:10])
        for data in x:
            #print("data=", data)
            #class_prob = []
            #for i in range(5): # n_class = 5
            class_prob = self.calculate_prob(data)
            #print("test=",class_prob)
            y_pred.append(np.argmax(np.array(class_prob) ))
        #print("y_pred=", y_pred)
        return y_pred

# TODO: change GaussianNB() to MyGaussianNB() once you finish the above code
data = []
#print(type(x))
#print(type(y))
#train_x = train_x.tolist()
#y= y.tolist()
#print(x)
#print(y)
'''
for ab in zip(train_x, train_y):
    temp =[]
    for a in ab:
        if isinstance(a,list):
            temp = a
        else:
            #print(a)
            #print(temp)
            temp.append(a)
            #print(temp)
    data.append(temp)
'''
gnb = MyGaussianNB()
#separated = gnb.split_dataset(data)
#gnb.compute_gaussian_params(separated)
gnb.fit(train_x, train_y)
#plot_decision_boundary(gnb, train_x, train_y)
pro = gnb.predict(test_x)
print("pro=",pro)
print(accuracy_score(test_y, pro))