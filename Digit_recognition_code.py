# Digit_recognition
##Import Libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
##Load MNIST digits dataset and observe the data
digits = datasets.load_digits()
print(digits)
print(digits.data)
print(digits.target)
print(digits.images[0])
##Use SVM as the classifier and split dataset into inputs(x) and targets(y) keeping last data-point for test
clf = svm.SVC(gamma = 0.001, C = 100)
print(len(digits.data))
x,y = digits.data[:-1],digits.target[:-1]
clf.fit(x,y)
##Predict Output of SVM classifier on test data-point and visualize the same for reference 
print("Prediction:", clf.predict(digits.data[-1]))
plt.imshow(digits.images[-1],cmap = plt.cm.gray_r,interpolation = "nearest")
plt.show()
