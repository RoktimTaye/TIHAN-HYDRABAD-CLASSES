import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification  ,make_moons,make_circles,make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

np.random.seed(7)

#Normal Prcptrom simulation

x,y = make_classification(n_samples=400,n_features=2,n_redundant=0,n_informative=2,n_clusters_per_class=1, class_sep=2.0,random_state=7)
xtr, xte,ytr,yte = train_test_split(x,y,test_size=0.2,random_state=7)

clf = Perceptron(max_iter=1000,random_state=7,eta0=0.1,tol=1e-5)
clf.fit(xtr,ytr)
pred = clf.predict(xte)
acc = accuracy_score(yte,pred)
print(f"Accuracy : {acc:.3f}")

h = 0.02
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Perceptron Classifier')
plt.show()

#Non-Lineear (Moon)

xm,ym = make_moons(n_samples=400,noise=0.2,random_state=7)

xm_tr,xm_te,ym_tr,ym_te = train_test_split(xm,ym,test_size=0.2,random_state=7)

p_moon = Perceptron(max_iter=1000,random_state=7,eta0=0.1,tol=1e-5)
p_moon.fit(xm_tr,ym_tr)
pred_moon = p_moon.predict(xm_te)
acc_moon = accuracy_score(ym_te,pred_moon)

h=0.02

x_min,y_min = x[:,0].min()-1,x[:,1].min()-1
x_max,y_max = x[:,0].max()+1,x[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx,yy,Z,alpha=0.3)
plt.scatter(x[:,0],x[:,1],c=y,alpha=0.8)
plt.title('Perceptron Classifier (Moon)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

xs = np.linspace(-7,7,400)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1-tanh(x)**2

def relu(x):
    return np.maximum(0,x)

def drelu(x):
    return np.where(x>0,1,0)

plt.figure()
plt.plot(xs, sigmoid(xs))
plt.title("Sigmoid")
plt.xlabel("x"); plt.ylabel("σ(x)")
plt.show()

plt.figure()
plt.plot(xs, dsigmoid(xs))
plt.title("Sigmoid Derivative")
plt.xlabel("x"); plt.ylabel("σ'(x)")
plt.show()

plt.figure()
plt.plot(xs, tanh(xs))
plt.title("Tanh")
plt.xlabel("x"); plt.ylabel("σ'(x)")
plt.show()

plt.figure()
plt.plot(xs, dtanh(xs))
plt.title("Tanh Derivative")
plt.xlabel("x"); plt.ylabel("σ'(x)")
plt.show()

plt.figure()
plt.plot(xs, relu(xs))
plt.title("relu")
plt.xlabel("x"); plt.ylabel("σ'(x)")
plt.show()

rng = np.random.default_rng(7)

def expected_grad_scale(activation, sample=20000):

    if activation == "relu":
        z = np.random.normal(0,np.sqrt(16),sample)
        g = drelu(z)
    elif activation == "sigmoid":
        z = np.random.normal(0,1,sample)
        g = dsigmoid(x)
    elif activation == "tanh":
        z = np.random.normal(0,10,sample)
        g = dtanh(z)

    else:
        raise ValueError("Unlknown activation")
    return np.mean(np.abs(g))

depths = np.arange(1,51)
scales = {a: [] for a in ["sigmoid","tanh","relu"]}

base = {a: expected_grad_scale(a) for a in scales}

for d in depths:
    for a in scales:
        scales[a].append(base[a]**d)

for a in scales:
    plt.figure()
    plt.plot(depths, scales[a])
    plt.title(f"Expected Gradient Scale vs Depth — {a}")
    plt.xlabel("Depth (layers)"); plt.ylabel("Relative scale")
    plt.yscale("log")
    plt.show()