CONTENTS:
optimizeAutoencoderLBFGS.m : Entry to the optimization program. This will call:
->  loadData.m : to load the smaller_dataset.mat (each column is an example)
->  initializeWeights.m : to initialize parameters similar to Glorot and Bengio, 2010.
->  minFunc.m (need to download from: http://www.di.ens.fr/~mschmidt/Software/minFunc_2009.zip) which will call
    ->  deepAutoencoder.m : the cost and gradient computation of variable depth and sized autoencoder
->  writeToTextFiles.m : write weights and biases to text format

Other files:    
checkGradient.m : check the gradient of the deepAutoencoder
extractFeatures.m : use Autoencoder features for other tasks



Please report bugs etc to Quoc V. Le (quocle@stanford.edu)

Related paper:
Q.V. Le, J. Ngiam, A. Coates, A. Lahiri, B. Prochnow, A.Y. Ng
On optimization methods for deep learning
ICML, 2011.
http://ai.stanford.edu/~quocle/LeNgiCoaLahProNg11.pdf

minFunc can be downloaded from Mark Schmidt's website:
http://www.di.ens.fr/~mschmidt/Software/minFunc_2009.zip
