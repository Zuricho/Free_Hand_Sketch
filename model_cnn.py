# import packages
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torchvision
import torch.utils.data.dataloader as DataLoader
from torchvision import transforms

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# training set
data_path_train = 'dataset/train'
train_dataset = torchvision.datasets.ImageFolder(
    root=data_path_train,
    transform=transform
)
train_loader = DataLoader.DataLoader(
    train_dataset,
    batch_size=100,
    num_workers=1,
    shuffle=True
)


# test set
data_path_test = 'dataset/test'
test_dataset = torchvision.datasets.ImageFolder(
    root=data_path_test,
    transform=transform
)
test_loader = DataLoader.DataLoader(
    test_dataset,
    batch_size=100,
    num_workers=1,
    shuffle=False
)

# validation set
data_path_valid = 'dataset/valid'
valid_dataset = torchvision.datasets.ImageFolder(
    root=data_path_valid,
    transform=transform
)
valid_loader = DataLoader.DataLoader(
    valid_dataset,
    batch_size=100,
    num_workers=1,
    shuffle=False
)


# convolutional neural network
import torch.nn as nn 
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self):
        # CNN model
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
        self.conv3 = nn.Conv2d(3, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3*10*10, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, len(train_dataset.classes))
        self.softmax = nn.Softmax()
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3*10*10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


# define a function to help get prediction
def make_prediction(loader, model):
    result_total = []
    reference_total = []
    for index, (data, target) in enumerate(loader):
        data, label = data.to(device), torch.eye(len(train_dataset.classes))[target].to(device)
        output = model(data)

        result = torch.max(output,dim=1).indices.cpu().detach().numpy()
        reference = torch.max(label,dim=1).indices.cpu().detach().numpy()

        result_total.append(result)
        reference_total.append(reference)
    return np.hstack(result_total), np.hstack(reference_total)


# training
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001 # learning rate
optimizer = optim.Adam(model.parameters(), lr=lr)
epoch_num = 1

loss_list = []

for epoch in range(1,epoch_num+1):
    # training process
    for batch_idx, (data, target) in enumerate(train_loader):
        data, label = data.to(device), torch.eye(len(train_dataset.classes))[target].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # calculate accuracy and other metrics
    torch.save(model.state_dict(), "./model/CNN_parameter_%d.pkl"%epoch)

    # plot the loss figure
    plt.figure(figsize=(12,4))
    plt.plot(loss_list,linewidth=1)
    plt.xlim(0,epoch*len(train_loader))
    plt.xlabel('epoch',fontsize=14)
    plt.ylabel('loss',fontsize=14)
    plt.xticks(np.arange(0,epoch*len(train_loader)+1,len(train_loader)),np.arange(0,epoch+1,1),fontsize=12)
    plt.savefig('./figure/CNN_loss_epoch_%d.png'%epoch)
    plt.close()

    # validation and test process
    result_valid, reference_valid = make_prediction(valid_loader, model)
    result_test, reference_test = make_prediction(test_loader, model)

    # calculate accuracy and other metrics
    print('Validation accuracy: %.4f'%accuracy_score(reference_valid, result_valid))
    print('Test accuracy: %.4f'%accuracy_score(reference_test, result_test))
    print('Validation F1 score: %.4f'%f1_score(reference_valid, result_valid, average='macro'))
    print('Test F1 score: %.4f'%f1_score(reference_test, result_test, average='macro'))
    print('Validation confusion matrix: \n%s'%confusion_matrix(reference_valid, result_valid))
    print('Test confusion matrix: \n%s'%confusion_matrix(reference_test, result_test))
    print('Validation classification report: \n%s'%classification_report(reference_valid, result_valid))
    print('Test classification report: \n%s'%classification_report(reference_test, result_test))

    
    plt.figure(figsize=(6,6))
    plt.title('Validation confusion matrix epoch %d'%epoch,fontsize=16)
    confusion_data = confusion_matrix(reference_valid, result_valid)
    plt.imshow(confusion_data,interpolation='nearest',cmap="YlGnBu",vmax=2500,vmin=0)
    for i in range(confusion_data.shape[0]):
        for j in range(confusion_data.shape[1]):
            plt.text(j,i,confusion_data[i,j],ha="center",va="center",fontsize=12)
    plt.xticks(np.arange(0,confusion_data.shape[1],1),fontsize=12)
    plt.yticks(np.arange(0,confusion_data.shape[0],1),fontsize=12)
    plt.savefig('./figure/CNN_valid_confusion_matrix_epoch_%d.png'%epoch)
    plt.close()


    plt.figure(figsize=(6,6))
    plt.title('Test confusion matrix epoch %d'%epoch,fontsize=16)
    confusion_data = confusion_matrix(reference_test, result_test)
    plt.imshow(confusion_data,interpolation='nearest',cmap="YlGnBu",vmax=2500,vmin=0)
    for i in range(confusion_data.shape[0]):
        for j in range(confusion_data.shape[1]):
            plt.text(j,i,confusion_data[i,j],ha="center",va="center",fontsize=12)
    plt.xticks(np.arange(0,confusion_data.shape[1],1),fontsize=12)
    plt.yticks(np.arange(0,confusion_data.shape[0],1),fontsize=12)
    plt.savefig('./figure/CNN_test_confusion_matrix_epoch_%d.png'%epoch)
    plt.close()


    
