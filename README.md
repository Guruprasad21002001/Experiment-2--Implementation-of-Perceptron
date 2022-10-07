# Experiment-2--Implementation-of-Perceptron
## AIM:
To implement a perceptron for classification using Python

## EQUIPMENTS REQUIRED:

Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one 
f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’

A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.


## ALGORITHM:

## Step 1:
Importing the libraries.

## Step 2:
Importing the dataset.

## Step 3:
Plot the data to verify the linear separable dataset and consider only two classes.

## Step 4:
Convert the data set to scale the data to uniform range by using Feature scaling.

## Step 5:
Split the dataset for training and testing.

## Step 6:
Define the input vector ‘X’ from the training dataset.

## Step 7:
Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2.

## Step 8:
Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’.

## Step 9:
Assign the learning rate.

## Step 10:
For ‘N ‘ iterations ,do the following: v(i) = w(i)*x(i)
~~~
    W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i).
~~~

## Step 11:
Plot the error for each iteration.

# Step 12:
Print the accuracy.

## Step 13:
End the program.

## PROGRAM:

### Developed by: GURUPRASAD.B
### Reg.No: 212221230032
Program to implement a perceptron for classification using Python.
~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron:
  def __init__(self, learning_rate=0.1):
    self.learning_rate = learning_rate
    self._b = 0.0
    self._w = None
    self.misclassified_samples = []
  def fit(self, x: np.array, y: np.array, n_iter=10):
    self._b = 0.0
    self._w = np.zeros(x.shape[1])
    self.misclassified_samples = []
    for _ in range(n_iter):
      errors = 0
      for xi, yi in zip(x, y):
        update = self.learning_rate * (yi - self.predict(xi))
        self._b += update
        self._w += update * xi
        errors += int(update != 0.0)
      self.misclassified_samples.append(errors)
  def f(self, x: np.array) -> float:
    return np.dot(x, self._w) + self._b
  def predict(self, x: np.array):
    return np.where(self.f(x) >= 0, 1, -1)

df = pd.read_csv('/content/IRIS.csv')
df.head()

y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values

x = x[0:100, 0:2]
y = y[0:100]

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
            label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()

y = np.where(y == 'Iris-setosa', 1, -1)

x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()

x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

classifier = Perceptron(learning_rate=0.01)

classifier.fit(x_train, y_train)

print("accuracy" , accuracy_score(classifier.predict(x_test), y_test)*100)

plt.plot(range(1, len(classifier.misclassified_samples) + 1),classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()
~~~
## OUTPUT:
## DF.HEAD():

![op1](https://user-images.githubusercontent.com/95342910/194471068-f26883bb-f5c5-4d4a-893e-bcb83fc23f2e.png)

## SCATTER PLOT ( SENTOSA Vs. VERSICOLOUR ):

![op2](https://user-images.githubusercontent.com/95342910/194471134-8aa47ef7-5360-40b6-b6f4-5c63e3a763b8.png)

## PLOT OF EPOCHS Vs. ERRORS AFTER CLASSIFICATION:

![op3](https://user-images.githubusercontent.com/95342910/194471220-a3947bec-a6f7-423f-aa6c-329729258280.png)

## ACCURACY:

![op4](https://user-images.githubusercontent.com/95342910/194471274-082a0a0d-abfa-489b-a5a7-3834c1f96ad1.png)

## RESULT:
Thus, to implement a perceptron for classification using Python is successfully executted.
