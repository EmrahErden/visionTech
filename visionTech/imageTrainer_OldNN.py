import glob         # required for reading folders
import numpy as np
import cv2
import theano
import theano.tensor as T
from theano import function
from theano import shared

# input parameters
folderName_and_Type = "32examples/*.jpg"
imageSize = 32
numExamples = 1263
pTrain = 0.8    # percent train
pTest = 0.15     # percent test, remainder is pVal
numOutputTypes = 2

# training parameters
initM = 0.1
ls1 = 20    #layer size 1
ls2 = 20    #layer size 2
learningRate = 0.1
numIterations = 10000
batchSize = 128
displayPer=100

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

nump = (int)(imageSize*imageSize) # number of pixels
XY_all=np.zeros(shape=(numExamples, nump+numOutputTypes))
imageFilesList = glob.glob(folderName_and_Type)
#print imageFilesList

fileCounter = 0;

#reading images from the folder
for file in imageFilesList:
    # Building up the X matrix
    image = cv2.imread(file)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayVector = grayImage.reshape((1, nump))
    XY_all[fileCounter, 0:nump] = grayVector/255.

    # Building up the Y matrix
    # finding the position of last /
    nameLength = len(file)
    slashPositions = [pos for pos, char in enumerate(file) if char == '/']
    lastSlashPosition = slashPositions[-1]
    outputName = ''
    for i in range(lastSlashPosition+1, nameLength):
        char = file[i]
        if (char != '_'):
            outputName += char
        else:
            break

    # EDIT HERE (later)!!
    XY_all[fileCounter, nump:] = 0
    if outputName == 'groundWheatGood':
        XY_all[fileCounter, nump] = 1.
    elif outputName == 'lentilGood':
        XY_all[fileCounter, nump+1] = 1.

    fileCounter += 1

nTrain = pTrain*numExamples
nTest = pTest*numExamples

#shuffling the big matrix in the end to for Xtrain, Ytrain, Xtest, Ytest
np.random.shuffle(XY_all)

Xtrain = XY_all[0:nTrain, 0:nump]
Ytrain = XY_all[0:nTrain, nump:]
Xtest = XY_all[nTrain:nTrain+nTest, 0:nump]
Ytest = XY_all[nTrain:nTrain+nTest, nump:]
Xval = XY_all[nTrain+nTest:, 0:nump]
Yval = XY_all[nTrain+nTest:, nump:]

"""
print Xtrain.shape
print Ytrain.shape
print Xtrain[0:1]
print Ytrain[0:1]
"""
#stop, train time!!
#symbolic math
X=T.fmatrix('X')Y=T.fmatrix('Y')
#declare shared variables
w1 = init_weights((nump, ls1))
w2 = init_weights((ls1, ls2))
w3 = init_weights((ls2, numOutputTypes))
#train
h1=T.nnet.sigmoid(T.dot(X, w1)) #https://github.com/Newmu/Theano-Tutorials/blob/master/3_net.py
h2=T.nnet.sigmoid(T.dot(h1, w2))
h3=T.nnet.softmax(T.dot(h2, w3))
pred = T.argmax(h3, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(h3, Y))

gw1, gw2, gw3 = T.grad(cost=cost, wrt=[w1, w2, w3])
train = theano.function(inputs=[X, Y], outputs = cost, updates=((w1, w1-learningRate*gw1), (w2, w2-learningRate*gw2), (w3, w3-learningRate*gw3)), allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs = pred, allow_input_downcast=True)

"""
print Xtrain.shape[0]

Yt = Ytrain[0:128]
Xt = Xtrain[0:128]
print Yt.shape
print Xt.shape
print len(Xt)

"""

for iteration in range(numIterations):
    for start, end in zip(range(0, Xtrain.shape[0], batchSize), range(batchSize, Xtrain.shape[0], batchSize)):
        Acost = train(Xtrain[start:end], Ytrain[start:end])
        Etrain = 1.0-np.mean(np.argmax(Ytrain, axis=1) == predict(Xtrain))
        Etest = 1.0-np.mean(np.argmax(Ytest, axis=1) == predict(Xtest))

    if(iteration%displayPer==0):
        print "iteration : " +str(iteration) + ", cost = " + str(Acost) +", Etrain = " + str(Etrain) + ", Etest = " + str(Etest)