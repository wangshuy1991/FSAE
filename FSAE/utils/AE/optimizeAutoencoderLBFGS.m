function [] = optimizeAutoencoderLBFGS(layersizes, datasetpath, ...
                                       finalObjective)
% train a deep autoencoder with variable hidden sizes
% layersizes : the sizes of the hidden layers. For istance, specifying layersizes =
%     [200 100] will create a network looks like input -> 200 -> 100 -> 200
%     -> output (same size as input). Notice the mirroring structure of the
%     autoencoders. Default layersizes = [2*3072 100]
% datasetpath: the path to the CIFAR dataset (where we find the *.mat
%     files). see loadData.m
% finalObjective: the final objective that you use to compare to
%                 terminate your optimization. To qualify, the objective
%                 function on the entire training set must be below this
%                 value.
%
% Author: Quoc V. Le (quocle@stanford.edu)
% 
%% Handle default parameters
if nargin < 3 || isempty(finalObjective)
    finalObjective = 70; % i am just making this up, the evaluation objective 
                         % will be much lower
end
if nargin < 2 || isempty(datasetpath)
  datasetpath = '.';
end
if nargin < 1 || isempty(layersizes)
  layersizes = [2*3072 100];
end

%% Load data
loadData

%% Random initialization
initializeWeights;

%% Optimization: minibatch L-BFGS
% Q.V. Le, J. Ngiam, A. Coates, A. Lahiri, B. Prochnow, A.Y. Ng. 
% On optimization methods for deep learning. ICML, 2011

addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 20;	  
options.display = 'on';
options.TolX = 1e-3;

perm = randperm(size(traindata,2));
traindata = traindata(:,perm);
batchSize = 1000;
maxIter = 20;
for i=1:maxIter    
    startIndex = mod((i-1) * batchSize, size(traindata,2)) + 1;
    fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
    data = traindata(:, startIndex:startIndex + batchSize-1); 
    [theta, obj] = minFunc( @deepAutoencoder, theta, options, layersizes, ...
                            data);
    if obj <= finalObjective % use the minibatch obj as a heuristic for stopping
                             % because checking the entire dataset is very
                             % expensive
        % yes, we should check the objective for the entire training set        
        trainError = deepAutoencoder(theta, layersizes, traindata);
        if trainError <= finalObjective
            % now your submission is qualified
            break
        end
    end
end

%% write to text files so that we can test your program
writeToTextFiles;