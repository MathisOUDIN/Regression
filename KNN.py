import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn import neighbors
from sklearn.model_selection import train_test_split

mnist = fetch_mldata('MNIST original')

print mnist.data.shape
print mnist.target.shape

sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.8)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

error = 1 - knn.score(X_test, y_test)
print('Erreur: %f' % error)

errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(X_train, y_train).score(X_test, y_test)))
plt.plot(range(2,15), errors, 'o-')
plt.show()

knn = neighbors.KNeighborsClassifier(4)
knn.fit(X_train, y_train)

predicted = knn.predict(X_test)

images = X_test.reshape((-1, 28, 28))

select = np.random.randint(images.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % predicted[value])

plt.show()

misclass = (y_test != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]

select = np.random.randint(misclass_images.shape[0], size=12)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: %i' % misclass_predicted[value])

plt.show()

