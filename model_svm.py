# import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torchvision
import torch.utils.data.dataloader as DataLoader
from torchvision import transforms

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])

# training set
data_path_train = 'dataset_pixel/train'
train_x = []
train_y = []
train_dataset = torchvision.datasets.ImageFolder(root=data_path_train,transform=transform)
for image_data in train_dataset:
    train_x.append(image_data[0].detach().numpy()[0].reshape(28*28))
    train_y.append(image_data[1])



# test set
data_path_test = 'dataset_pixel/test'
test_x = []
test_y = []
test_dataset = torchvision.datasets.ImageFolder(root=data_path_test,transform=transform)
for image_data in test_dataset:
    test_x.append(image_data[0].detach().numpy()[0].reshape(28*28))
    test_y.append(image_data[1])

# validation set
data_path_valid = 'dataset_pixel/valid'
valid_x = []
valid_y = []
valid_dataset = torchvision.datasets.ImageFolder(root=data_path_valid,transform=transform)
for image_data in valid_dataset:
    valid_x.append(image_data[0].detach().numpy()[0].reshape(28*28))
    valid_y.append(image_data[1])



# Start build SVM
classifier = SVC(kernel="poly")
classifier.fit(train_x, train_y)


print("test set result")
predict_y = classifier.predict(test_x)
print(confusion_matrix(test_y, predict_y))
print(classification_report(test_y, predict_y))
print(accuracy_score(test_y, predict_y))
print(f1_score(test_y, predict_y, average='macro'))

print("validation set result")
predict_y = classifier.predict(valid_x)
print(confusion_matrix(valid_y, predict_y))
print(classification_report(valid_y, predict_y))
print(accuracy_score(valid_y, predict_y))
print(f1_score(valid_y, predict_y, average='macro'))
