import numpy as np
import matplotlib.pyplot as plt
TrainDigits = np.load("HandwrittenDigits/TrainDigits.npy")

SVDER = np.zeros()
for i in range(10):
    d = TrainDigits[:,i] # The first digit in the training set
    D = np.reshape(d, (28, 28)).T # Reshaping a vector to a matrix
    np.linalg.svd(D)
    plt.imshow(D, cmap ="gray") # Plot of the digit       
    #plt.show()

TrainLabels = np.load("HandwrittenDigits/TrainLabels.npy")

print(TrainLabels[:,0])
