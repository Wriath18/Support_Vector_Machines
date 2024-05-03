# %%
# !pip install mlp

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles
import numpy as np

# %%

X,Y = make_circles(n_samples=500, noise=0.02)

# %%
plt.scatter(X[:,0], X[:,1], c=Y)

# %%
def phi(X):
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X1**2 + X2**2

    X_ = np.zeros((X.shape[0], 3))
    X_[:,:-1] = X
    X_[:,-1] = X3

    return X_


# %%
X_ = phi(X)
print(X_.shape)

# %%
def plot3d(X):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]

    ax.scatter(X1,X2,X3 , zdir='z', s=20, c=Y, depthshade=True)

    return ax

# %%
plot3d(X_)

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# %%
lr = LogisticRegression()


# %% [markdown]
# Cross validating the original dataset

# %%
score = cross_val_score(lr,X,Y,cv=5).mean()
print(score*100)

# %% [markdown]
# Cross validation on the Generated high dimention dataset

# %%
score = cross_val_score(lr, X_, Y, cv=5).mean()
print(score*100)

# %%
lr.fit(X_,Y)

# %%
weights = lr.coef_



# %%
bias = lr.intercept_

# %% [markdown]
# Making a imaginary meshgrip to measure z as this is a 3d plot so the equation will be w1x1 + w2x2 + w3x3+ bias = 0

# %%
xx, yy = np.meshgrid(range(-2,2), range(-2,2))

# %%
z = -(weights[0,0]*xx + weights[0,1]*yy + bias)/weights[0,2]

# %%
z

# %%
ax = plot3d(X_)
ax.plot_surface(xx,yy,z)
plt.show()

# %%



