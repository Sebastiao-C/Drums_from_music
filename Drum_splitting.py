# %%
import sys, os, os.path

#!{sys.executable} -m pip install torch
#!{sys.executable} -m pip install pandas


from scipy.io import wavfile
import pandas as pd




import torch
import torch.nn as nn


# %%

#input_filename = input("Input file name:")
#if input_filename[-3:] != 'wav':
#    print('WARNING!! Input File format should be *.wav')
#    sys.exit()

samrate, data = wavfile.read('./mixture.wav')
samrate_Drums, data_Drums = wavfile.read('./drums.wav')

# %%
print(len(data))
print(samrate)

# %%
len(data_Drums)/samrate_Drums/60

# %%
len(data)/samrate/60

# %%
#sum(data[:,0] == data[:,1])

# %%
#wavfile.write(data=(data - data_Drums), filename='./nodrums.wav', rate=samrate)


# %%
import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt





# %%

class DrumsSplitter(nn.Module):
    def __init__(
        self
        
    ):
        super(DrumsSplitter, self).__init__()
        #self.conv1 = nn.Conv2d(1, 100, 5)
        #self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(2)
        #self.conv2 = nn.Conv2d(100, 20, 5)        
        
        self.conv1 = nn.Conv2d(1, 10, (6,7))
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(10, 20, (5,6))
        self.conv3 = nn.Conv2d(20, 40, (4,6))
        self.conv4 = nn.Conv2d(40, 80, (4,6))
        self.conv5 = nn.Conv2d(80, 120, (4,5))
        self.upsamp = nn.Upsample(scale_factor=(2,2))#, output_size=)
        self.convt5 = nn.ConvTranspose2d(120, 80, (4,5))
        self.convt4 = nn.ConvTranspose2d(80, 40, (4,6))
        self.convt3 = nn.ConvTranspose2d(40, 20, (4,6))
        self.convt2 = nn.ConvTranspose2d(20, 10, (5,6))
        self.convt1 = nn.ConvTranspose2d(10, 1, (6,7))
        self.linear = nn.Linear( 1* 129* 492, 1* 129* 492)
        
    def forward(
        self,
        x
    ):
        X = self.conv1(x)
        #print(X.shape)
        X = self.relu(X)
        X = self.maxpool(X)        
        #print(X.shape)
        X = self.conv2(X)
        #print(X.shape)
        X = self.relu(X)
        X = self.maxpool(X)
        #print(X.shape)
        X = self.conv3(X)
        #print(X.shape)
        X = self.relu(X)
        X = self.maxpool(X)
        #print(X.shape)
        X = self.conv4(X)
        #print(X.shape)
        X = self.relu(X)
        X = self.maxpool(X)
        #print(X.shape)
        X = self.conv5(X)
        #print(X.shape)
        X = self.relu(X)
        X = self.maxpool(X)
        #print(X.shape)
        X = self.upsamp(X)
        #print(X.shape)
        X = self.convt5(X, output_size=(8, 200, 5, 26))
        X = self.relu(X)
        X = self.upsamp(X)
        X = self.convt4(X, output_size=(8, 100, 13, 57))
        X = self.relu(X)
        X = self.upsamp(X)
        X = self.convt3(X, output_size=(8, 50, 29, 119))
        X = self.relu(X)
        X = self.upsamp(X)
        X = self.convt2(X, output_size=(8, 20, 62, 243))
        X = self.relu(X)
        X = self.upsamp(X)
        X = self.convt1(X, output_size=(8, 1, 129, 492))
        X = self.relu(X)
        #print("final shape" + str( X.shape))

        X = self.linear(X)
        return X


# %%
def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """


    # clear the gradients
    optimizer.zero_grad()
    # compute the model output
    yhat = model(X)
    # calculate loss
    loss = criterion(yhat, y)
    # credit assignment
    loss.backward() # computes the gradients.
    optimizer.step() # updates weights using the gradients.

    return loss.item()

# %%

def predict(model, X):
    """X (n_examples x n_features)"""
    #print(X[0])
    scores = model(X)  # (n_examples x n_classes)
    #predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    #print(scores[0])
    return scores


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    #print("before pred")

    y_hat = predict(model, X)
    #print("after pred")
    #n_correct = (y == y_hat).sum().item()
    #n_possible = float(y.shape[0])
    #print("before loss")
    #print(y_hat.shape)
    #print(y.shape)
    err = nn.functional.mse_loss(y, y_hat)
    #print(err.shape)
    #print("after loss")

    model.train()
    return err


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')
    plt.show()

# %%
import os
# assign directory
directory = './train'
 
# iterate over files in
# that directory
data_train = []
drums_train = []
for folder in os.listdir(directory):
    f = os.path.join(directory, folder)
    # checking if it is a file
    for filename in os.listdir(f):
        f2 = os.path.join(f, filename)
        if os.path.isfile(f2) and "drums" in f2 and "Identifier" not in f2:

            samrate_Drums, data_Drums = wavfile.read(f2)
            drums_train.append(data_Drums)
        elif os.path.isfile(f2) and "mixture" in f2 and "Identifier" not in f2:
            samrate, data = wavfile.read(f2)
            data_train.append(data)
            


# %%
def splitIntoOverlapping10s(array):
    leng = len(array)
    splits = []
    currsize = 0
    while(True):
        if currsize + 110250 < leng:
            splits.append(array[currsize:currsize+110250])
            currsize += 220500 
        else:
            break
    return splits

print("here1")

# %%
x_train = []

for song in data_train:
    splits = splitIntoOverlapping10s(song)
    for split in splits:
        x_train.append(split)



X_train = np.array(x_train)


# %%
len(drums_train[0])

print("here2")

# %%
y_train = []

for song in drums_train:
    splits = splitIntoOverlapping10s(song)
    #splits = np.array(splits)
    for split in splits:
        y_train.append(split)



# %%
Y_train = np.array(y_train)


# %%
Y_train.shape == X_train.shape
X_train.shape

# %%
XX_train = X_train[:,:,0]
YY_train = Y_train[:,:,0]

x_train = list(XX_train)
y_train = list(YY_train)

del XX_train
del YY_train



# %%
XXs = []
count = 0
for snippet in range(len(x_train)):
    if count % 1000 == 0:
        print(count)
    if count == 4000:
        break
    f, t, Sxx = signal.spectrogram(x_train[snippet], samrate)
    XXs.append(Sxx)
    count += 1

print(len(XXs))
print(len(XXs[0]))
print(len(XXs[0][0]))

YYs = []
count = 0
for snippet in range(len(y_train)):
    if count % 1000 == 0:
        print(count)
    if count == 4000:
        break
    f, t, Sxx = signal.spectrogram(y_train[snippet], samrate)
    YYs.append(Sxx)
    count += 1

print(len(YYs))
print(len(YYs[0]))
print(len(YYs[0][0]))

Xs = torch.tensor(XXs).unsqueeze(1).unsqueeze(1)
Ys = torch.tensor(YYs).unsqueeze(1).unsqueeze(1)

del XXs
del YYs

directory = './test'

data_val = []
drums_val = []
for folder in os.listdir(directory):
    f = os.path.join(directory, folder)
    # checking if it is a file
    for filename in os.listdir(f):
        f2 = os.path.join(f, filename)
        if os.path.isfile(f2) and "drums" in f2 and "Identifier" not in f2 and "Happy" not in f2 and "Oh No" not in f2:
            #print(f2)
            samrate_Drums, data_Drums = wavfile.read(f2)
            drums_val.append(data_Drums)
        elif os.path.isfile(f2) and "mixture" in f2 and "Identifier" not in f2 and "Happy" not in f2 and "Oh No" not in f2:
            #print(f2)
            samrate, data = wavfile.read(f2)
            data_val.append(data)
            



print("here1")

# %%
x_val = []

for song in data_val:
    splits = splitIntoOverlapping10s(song)
    for split in splits:
        x_val.append(split)



X_val = np.array(x_val)


# %%
len(drums_val[0])

print("here2")

# %%
y_val = []

for song in drums_val:
    splits = splitIntoOverlapping10s(song)
    #splits = np.array(splits)
    for split in splits:
        y_val.append(split)



# %%
Y_val = np.array(y_val)


# %%
Y_val.shape == X_val.shape
X_val.shape

# %%
XX_val = X_val[:,:,0]
YY_val = Y_val[:,:,0]

x_val = list(XX_val)
y_val = list(YY_val)

del XX_val
del YY_val


# %%
Xvals = []
count = 0
for snippet in range(len(x_val)):
    if count % 100 == 0:
        print(count)
    if count == 1000:
        break
    f, t, Sxx = signal.spectrogram(x_val[snippet], samrate)
    Xvals.append(Sxx)
    count += 1

print(len(Xvals))
print(len(Xvals[0]))
print(len(Xvals[0][0]))

Yvals = []
count = 0
for snippet in range(len(y_val)):
    if count % 100 == 0:
        print(count)
    if count == 1000:
        break
    f, t, Sxx = signal.spectrogram(y_val[snippet], samrate)
    Yvals.append(Sxx)
    count += 1

print(len(Yvals))
print(len(Yvals[0]))
print(len(Yvals[0][0]))

Xvals = torch.tensor(Xvals).unsqueeze(1)
Yvals = torch.tensor(Yvals).unsqueeze(1)













def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%

def go():

    #utils.configure_seed(seed=42)


    #data = utils.load_classification_data()
    #dataset = utils.ClassificationDataset(data)
    #train_dataloader = DataLoader(
    #    dataset, batch_size=opt.batch_size, shuffle=True)
    #dev_X, dev_y = dataset.dev_X, dataset.dev_y
    #test_X, test_y = dataset.test_X, dataset.test_y

    ## LOAD DATA ##
    #############################################################################################################3

    # initialize the model
    model = DrumsSplitter()
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims["adam"]
    optimizer = optim_cls(
        model.parameters(), lr=0.0002, weight_decay=0#, foreach=True
    )
    
    # get a loss criterion
    criterion = nn.MSELoss()
    
    # training loop
    epochs = np.arange(1, 30 + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []

    print(Xs.shape)
    print(Ys.shape)
    finalX = torch.stack(torch.split(Xs, 8), dim=0).squeeze(3)
    finalY = torch.stack(torch.split(Ys, 8), dim=0).squeeze(3)
    print(finalX.shape)
    print(finalY.shape)

    print(count_parameters(model))

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        #print(Xs[0])
        #print(Ys[0])
        for X_batch, y_batch in zip(finalX, finalY):
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        #print("before val")
        valid_accs.append(evaluate(model, Xvals, Yvals).detach().numpy())
        #print("after val")
        print('Valid acc: %.4f' % (valid_accs[-1]))

    #print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}".format(0.0002, 0, "adam")

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))

import gc

del data
del data_Drums
del data_train
del data_val
del drums_train
del drums_val


gc.collect()

go()

