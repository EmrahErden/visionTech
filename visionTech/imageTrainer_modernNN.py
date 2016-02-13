import glob         # required for reading folders
import numpy as np
import cv2
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# input parameters
folderName_and_Type = "32examples/*.jpg"
imageSize = 32
numExamples = 1263
pTrain = 0.8    # percent train
pTest = 0.15     # percent test, remainder is pVal
numOutputTypes = 2

# training parameters
initM = 0.1
ls1 = 40    #layer size 1
ls2 = 40    #layer size 2
learningRate = 0.001
numIterations = 5000
batchSize = 128
displayPer=10
srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w1, w2, w3, p_drop_input, p_drop_hidden):
    X = dropout(X, p_drop_input)
    h = rectify(T.dot(X, w1))

    h = dropout(h, p_drop_hidden)
    h2 = rectify(T.dot(h, w2))

    h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(T.dot(h2, w3))
    return h, h2, py_x

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


#stop, hammer-train time!!
#symbolic math
X=T.fmatrix('X')
Y=T.fmatrix('Y')
#declare shared variables
w1 = init_weights((nump, ls1))
w2 = init_weights((ls1, ls2))
w3 = init_weights((ls2, numOutputTypes))

noise_h, noise_h2, noise_py_x = model(X, w1, w2, w3, 0.2, 0.5)
h, h2, py_x = model(X, w1, w2, w3, 0., 0.)
y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w1, w2, w3]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for iteration in range(numIterations):
    for start, end in zip(range(0, Xtrain.shape[0], batchSize), range(batchSize, Xtrain.shape[0], batchSize)):
        Acost = train(Xtrain[start:end], Ytrain[start:end])
        Etrain = 1.0-np.mean(np.argmax(Ytrain, axis=1) == predict(Xtrain))
        Etest = 1.0-np.mean(np.argmax(Ytest, axis=1) == predict(Xtest))

    if(iteration%displayPer==0):
        print "iteration : " +str(iteration) + ", cost = " + str(Acost) +", Etrain = " + str(Etrain) + ", Etest = " + str(Etest)