# Restricted-Boltzmann-Machine-(RBM)

It is a implemetation of Restricted-Boltzmann-Machine in Tensorflow.

Restricted-Boltzmann-Machine (RBM) is a generative stochastic artificial neural network cosisting of two layers. Restricted Boltzmann Machines is used for Unsupervised Learning of Features based on a probabilistic model. RBMs are often trained using Cotrastive Divergence (CD) algorithm. Learning procedure of RBM consists of several steps of Gibbs sampling, i.e, samplig hidden units given visible units and vice-versa. The weights are then adjusted by using CD algorithm to minimize reconstruction error. 

The feature obtained after training on MNIST data after 20000 iteration with batch size of 10.

![My image](https://github.com/yasertaheri/Restricted-Boltzmann-Machine-RBM-/blob/master/Figure_1.png)

Test image :

![My image](https://github.com/yasertaheri/Restricted-Boltzmann-Machine-RBM-/blob/master/Figure_2.png)

Reconstructed image :

![My image](https://github.com/yasertaheri/Restricted-Boltzmann-Machine-RBM-/blob/master/Figure_3.png)



