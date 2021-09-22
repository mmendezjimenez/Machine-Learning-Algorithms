#Taking Example three and implementing it in pytorch

#Necessary Imports 

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from sklearn import datasets, linear_model

#Loading Diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y = True)

#Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

#Split the data into training/testing sets 
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

#Split the targets into training/testing sets
diabetes_Y_train = diabetes_y[:-20]
diabetes_Y_test = diabetes_y[-20:]

#Defining the model architecture
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

#Create the parameters for the model

inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 100        #Number of iterations the algorithm will run through

model = linearRegression(inputDim, outputDim)
#Makes use of the GPU if you have one available
if torch.cuda.is_available():
    model.cuda()

#Then we initialize the loss function and the optimizer

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    #Converting inputs and labels to Variables
    
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(diabetes_X_train).cuda())
        labels = Variable(torch.from_numpy(diabetes_Y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(diabetes_X_train))
        labels = Variable(torch.from_numpy(diabetes_Y_train))

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    print(loss)
    loss.backward()
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

#testing the model and plotting
with torch.no_grad():
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(diabetes_X_test).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(diabetes_X_test))).cpu().data.numpy()
    print(predicted)

plt.clf()
plt.plot(diabetes_X_train, diabetes_Y_train, 'go', label='True data', alpha=0.5)
plt.plot(diabetes_Y_test, predicted, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()