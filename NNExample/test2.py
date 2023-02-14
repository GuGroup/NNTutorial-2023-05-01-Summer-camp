# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:42:37 2023

@author: user
"""
# https://github.com/ECSIM/pem-dataset1/tree/master/Standard%20Test%20of%20Nafion%20Membrane%20112
# https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60c748624c8919b500ad2ef3/original/dataset-of-standard-tests-of-nafion-112-membrane-and-membrane-electrode-assembly-mea-activation-tests-of-proton-exchange-membrane-pem-fuel-cell.pdf
import csv
import torch

# load data
data = []
with open('data.csv') as f:
    reader = csv.reader(f)
    next(reader)
    data = [[float(ll) for ll in l] for l in reader]

data = torch.Tensor(data)

y = data[:,:1]
x = data[:,1:]

# divide data set in to train and test
randomized_idx = torch.randperm(data.shape[0])

train_ratio = 0.8
ntrain = int(data.shape[0]*train_ratio)

idx_train = randomized_idx[:ntrain]
idx_test = randomized_idx[ntrain:]

x_train = x[idx_train,:]
y_train = y[idx_train]

x_test  = x[idx_test,:]
y_test  = y[idx_test]

# normalize data

x_train_mean = x_train.mean(0)
x_train_std = x_train.std(0)

x_train_normalized = (x_train -x_train_mean)/x_train_std
x_test_normalized = (x_test -x_train_mean)/x_train_std

y_train_mean  = y_train.mean()
y_train_std  = y_train.std()
y_train_normalized = (y_train - y_train_mean)/y_train_std
y_test_normalized = (y_test - y_train_mean)/y_train_std


# model

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(4, 32),
                        torch.nn.Softplus(),
                        torch.nn.Linear(32, 32),
                        torch.nn.Softplus(),
                        torch.nn.Linear(32, 1))
        
    def forward(self, x):
        y = self.model(x)
        return y

model = NeuralNetwork()

# training
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),0.001)
for i in range(10000):
    y_train_normalized_pred = model(x_train_normalized)
    loss = criterion(y_train_normalized_pred,y_train_normalized)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(i,str(loss))

y_test_normalized_pred = model(x_test_normalized)
loss = criterion(y_test_normalized_pred,y_test_normalized)
print(loss)

y_test_pred = y_test_normalized_pred*y_train_std+y_train_mean
for i in range(y_test_pred.shape[0]):
    print(y_test_pred[i],y_test[i])

import matplotlib.pyplot as plt

pressure = 1
relative_humidity =100
membrane_compression = 18

# example_data = []
# for d in data:
#     if d[2] == pressure and d[3] == relative_humidity and d[4] ==membrane_compression:
#         example_data.append(d)
# example_data = torch.stack(example_data)

test_x_data = []
voltages = torch.linspace(0.0,1.0,100)
for v in voltages:
    test_x_data.append([v,pressure,relative_humidity,membrane_compression])
test_x_data = (torch.tensor(test_x_data) -x_train_mean)/x_train_std

test_y_data = model(test_x_data)
test_y_data = test_y_data.detach()
test_y_data = test_y_data*y_train_std+y_train_mean

plt.figure()
# plt.scatter(example_data[:,1],example_data[:,0])
plt.ylabel('current density [mA/cm2]')
plt.xlabel('voltage [V]')
plt.ylim([0,2200])
plt.xlim([0,1.1])
plt.title('%f %f %f'%(pressure,relative_humidity,membrane_compression))
plt.plot(voltages,test_y_data)
