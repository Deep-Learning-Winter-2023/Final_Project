from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch.optim as optim

import numpy as np

from onlyCNN import CNN

PATH_TO_SAVE = "./mel_cnn.mdl"

classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

images = np.load('./data/genres/mel_data/numerical_data/images.npy')
labels = np.load('./data/genres/mel_data/numerical_data/labels.npy')

images_train = np.zeros(shape=(800, 3, 100, 150))
images_test = np.zeros(shape=(200, 3, 100, 150))
labels_train = np.zeros(shape=(800))
labels_test = np.zeros(shape=(200))
citr = cite = cltr = clte = 0

for i, c in enumerate(classes):
    sub_perm = i*100 + np.random.RandomState(seed=20).permutation(100)
    for z in sub_perm[:80]:
        images_train[citr] = images[z]
        labels_train[cltr] = labels[z]
        citr += 1
        cltr += 1
    for z in sub_perm[80:]:
        images_test[cite] = images[z]
        labels_test[clte] = labels[z]
        cite += 1
        clte += 1


# yo = 0
# for i, l in enumerate(labels_train):
#     if l == 9: yo += 1

# print(yo)

images = torch.from_numpy(images).float()
labels = torch.from_numpy(labels).long()

images_train = torch.from_numpy(images_train).float()
images_test = torch.from_numpy(images_test).float()
labels_train = torch.from_numpy(labels_train).long()
labels_test = torch.from_numpy(labels_test).long()

cnn = CNN()                                         # defining the CNN object
print("load from saved model87013487 [y/n]?")
ans = input()
if ans == 'y':                                      # if saved model is to be used
    cnn.load_state_dict(torch.load(PATH_TO_SAVE))
    cnn.eval()


correct = 0
total = 0
with torch.no_grad():
    # for (images, labels) in test_loader:
    outputs = cnn(images_test)
    _, predicted = torch.max(outputs.data, 1)
    total += labels_test.size(0)
    correct += (predicted == labels_test).sum().item()

print('Accuracy of the network on the test images: %f %%' % (
    100 * (correct / total)))

print(classification_report(labels_test, predicted))
