import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fcnal
from sklearn.pipeline import Pipeline
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

# Determine device to run network on (runs on gpu if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_target_model(model=None, data_loader=None, classes=None):
    """
    Function to evaluate a target model provided
    specified data sets.
    Parameters
    ----------
    model       : Module
                  PyTorch conforming nn.Module function
    data_loader : DataLoader
                  PyTorch dataloader function
    classes     : list
                  list of classes
    Returns
    -------
    accuracy    : float
                  accuracy of target model
    """

    if classes is not None:
        n_classes = len(classes)
        class_correct = np.zeros(n_classes)
        class_total = np.zeros(n_classes)
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for i, (imgs, lbls) in enumerate(data_loader):

            imgs, lbls = imgs.to(device), lbls.to(device)

            output = model(imgs)

            predicted = output.argmax(dim=1)

            total += imgs.size(0)
            correct += predicted.eq(lbls).sum().item()

            if classes is not None:
                for prediction, lbl in zip(predicted, lbls):

                    class_correct[lbl] += prediction == lbl
                    class_total[lbl] += 1

    accuracy = 100*(correct/total)
    if classes is not None:
        for i in range(len(classes)):
            print('Accuracy of {} : {:.2f} %%'
                  .format(classes[i],
                          100 * class_correct[i] / class_total[i]))

    print("\nAccuracy = {:.2f} %%\n\n".format(accuracy))

    return accuracy

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Learner(nn.Module):
    def __init__(self):
        super(Learner, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


class softCrossEntropy(torch.nn.Module):
    def __init__(self):
        """
        :param alpha: Strength (0-1) of influence from soft labels in training
        """
        super(softCrossEntropy, self).__init__()
        self.alpha = 0.95
        return

    def forward(self, inputs, target, true_labels):
        """
        :param inputs: predictions
        :param target: target (soft) labels
        :param true_labels: true (hard) labels
        :return: loss
        """

        KD_loss = self.alpha
        KD_loss *= nn.KLDivLoss(size_average=False)(
                                 fcnal.log_softmax(inputs, dim=1),
                                 fcnal.softmax(target, dim=1)
                                )
        KD_loss += (1-self.alpha)*fcnal.cross_entropy(inputs, true_labels)

        return KD_loss


def distill_training(teacher=None, learner=None, data_loader=None,
                     test_loader=None, optimizer=None,
                     criterion=None, n_epochs=0, verbose=False):
    """
    :param teacher: network to provide soft labels in training
    :param learner: network to distill knowledge into
    :param data_loader: data loader for training data set
    :param test_loaderL data loader for validation data
    :param optimizer: optimizer for training
    :param criterion: objective function, should allow for soft labels.
                      We suggest softCrossEntropy
    :param n_epochs: epochs for training
    :param verbose: verbose == True will print loss at each batch
    :return: None, teacher model is trained in place
    """
    losses = []
    for epoch in range(n_epochs):
        teacher.eval()
        learner.train()
        for i, batch in enumerate(data_loader):
            with torch.set_grad_enabled(False):
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
                soft_lables = teacher(data)

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                outputs = learner(data)
                loss = criterion(outputs, soft_lables, labels)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if verbose:
                    print("[{}/{}][{}/{}] loss = {}"
                          .format(epoch, n_epochs, i,
                                  len(data_loader), loss.item()))

        # evaluate performance on testset at the end of each epoch
        print("[{}/{}]".format(epoch, n_epochs))

        print("Training:")
        train_acc = eval_target_model(learner, data_loader, classes=None)

        print("Testing:")
        test_acc = eval_target_model(learner, test_loader, classes=None)
    return train_acc, test_acc


batch_size=128
lr=1e-3
log_interval=100
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)) ])),batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./', train=False, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=batch_size, shuffle=True)


teacher = Teacher().to(device)
optimizer = optim.Adadelta(teacher.parameters(), lr=lr)
train(teacher, device, train_loader, optimizer,1)


learner = Learner().to(device)
optimizer = optim.Adadelta(learner.parameters(), lr=lr)
criterion=softCrossEntropy()

distill_training(teacher, learner, train_loader, test_loader, optimizer, criterion, 2)
