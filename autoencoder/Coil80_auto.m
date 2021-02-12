%%% demo code for point-wise gated Boltzmann machines (PGBM)
%   on variations of MNIST datasets.
%
%   the pipeline is given as follows:
%   1. load data
%   2. pretrain RBM
%   3. train PGBM with pretrained RBM as an initialization
%   4. evaluate the performance of PGBM using linear SVM (liblinear)
%   5. write the results on log file

clear all;
close all;
warning off
dataset = 'Coil';

path('liblinear-1.93/.',path);
% addpath(genpath('liblinear-2.1/.'));
addpath(genpath('../data/.'));
addpath(genpath('../FS/.'));
addpath(genpath('../autoencoder/.'));
addpath ./utils/minFunc/
data = load('../data/Coil/COIL100.mat');

fea=double(data.fea);
gnd=data.gnd;
fea = fea/255;

TestIdx=[];
TrainIdx=[];
for loop = [1,2,6,7,10,13,15,16,19,20,21:30]
    load(strcat('../data/Coil/10Train/',num2str(loop),'.mat'))
    trainIdx = trainIdx(1:end/5*4,:);
    testIdx = testIdx(1:end/5*4,:);
    TestIdx = [TestIdx testIdx];
    TrainIdx = [TrainIdx trainIdx];
end

rate_all = zeros(1,20);
rate_all_fs = zeros(1,20);
rate_all_fs2 = zeros(1,20);
options.Method = 'lbfgs';
options.maxIter = 20;
options.display = 'off';
options.TolX = 1e-7;
for FSratio = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
    for loop = 1:20
        xtrain = fea(TrainIdx(:,loop),:)';
        xtest = fea(TestIdx(:,loop),:)';
        ytrain = gnd(TrainIdx(:,loop));
        ytest = gnd(TestIdx(:,loop));
        
        [eigvec, eigval, ~, sampleMean] = PCA(xtrain',300);
        Wdims = size(eigvec, 2);
        xtrain = (bsxfun(@minus, xtrain', sampleMean) * eigvec(:, 1:Wdims))';
        xtest = (bsxfun(@minus, xtest', sampleMean) * eigvec(:, 1:Wdims))';
        
        xtrain = [xtrain*diag(sparse(1./sqrt(sum(xtrain.^2))))];
        xtest = [xtest*diag(sparse(1./sqrt(sum(xtest.^2))))];
        
        %         xtraintest = mapminmax([xtrain,xtest],-1,1);
        %         xtrain = xtraintest(:,1:size(ytrain));
        %         xtest = xtraintest(:,end-size(ytest)+1:end);
        
        %%% 1. load data
        %[xtrain, ytrain, xval, yval] = load_mnist(dataset);
        traindata = xtrain;
        trainlabel = ytrain;
        layersizes = 300;
        finalObjective = 70/50000;
        lambda1 = 0.002;
        lambda2 = 0.30;
        %FSratio = 0.5;
        [theta,P_fs] = FSAE(traindata,trainlabel,layersizes,options,finalObjective,lambda1,lambda2,FSratio);
        
        ztrain = extractFeatures(theta, layersizes, xtrain );
        %zval = extractFeatures(theta, layersizes, xval );
        ztest = extractFeatures(theta, layersizes, xtest );
        
        %%%%%%%%%%%%%%%%%%%%%%% stack 2-th layer %%%%%%%%%%%%%%%%%%%
        traindata = ztrain{1};
        trainlabel = ytrain;
        layersizes2 = 200;
        finalObjective = 70/50000;
        lambda3 = 0.002;
        lambda4 = 0.30;
        %FSratio = 0.5;
        [theta,P2_fs] = FSAE(traindata,trainlabel,layersizes2,options,finalObjective,lambda3,lambda4,FSratio);
        
        zztrain = extractFeatures(theta, layersizes2, ztrain{1} );
        %zval = extractFeatures(theta, layersizes, xval );
        zztest = extractFeatures(theta, layersizes2, ztest{1} );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
            %%%%%%%%%%%%%%%%%%%%%%% stack 3-th layer %%%%%%%%%%%%%%%%%%%
            traindata = zztrain{1};
            trainlabel = ytrain;
            layersizes3 = 100; %  together 9370 320 layer 3 9396 390
            finalObjective = 70/50000;
            lambda5 = 0.002;
            lambda6 = 0.3;
            %FSratio = 0.5;
            [theta,P3_fs] = FSAE(traindata,trainlabel,layersizes3,options,finalObjective,lambda5,lambda6,FSratio);
        
            zzztrain = extractFeatures(theta, layersizes3, zztrain{1} );
            %zval = extractFeatures(theta, layersizes, xval );
            zzztest = extractFeatures(theta, layersizes3, zztest{1} );
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %     %%%%%%%%%%%%%%%%%%%%%%% stack 4-th layer %%%%%%%%%%%%%%%%%%%
        %     traindata = zzztrain{1};
        %     trainlabel = ytrain;
        %     layersizes4 = 300; %  together 9370 320 layer 3 9396 390
        %     finalObjective = 70/50000;
        %     lambda7 = 0.0015;
        %     lambda8 = 0.75;
        %     FSratio = 0.5;
        %     [theta,P4_fs] = FSAE(traindata,trainlabel,layersizes4,options,finalObjective,lambda7,lambda8,FSratio);
        %
        %     zzzztrain = extractFeatures(theta, layersizes4, zzztrain{1} );
        %     %zval = extractFeatures(theta, layersizes, xval );
        %     zzzztest = extractFeatures(theta, layersizes4, zzztest{1} );
        %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        Cls = cvKnn(xtest, xtrain, ytrain', 1);
        acc = length(find(Cls==ytest'))/length(ytest);
        %fprintf('DAE+%0.4f\n',acc);
        
        Cls = cvKnn(ztest{1}, ztrain{1}, ytrain', 1);
        acc = length(find(Cls==ytest'))/length(ytest);
        %fprintf('DAE+%0.4f\n',acc);
        rate_all(1,loop) = acc;
        
        Cls = cvKnn(P_fs*ztest{1}, P_fs*ztrain{1}, ytrain', 1);
        acc = length(find(Cls==ytest'))/length(ytest);
        %fprintf('DAE+%0.4f\n',acc);
        
        Cls = cvKnn(P2_fs*zztest{1}, P2_fs*zztrain{1}, ytrain', 1);
        acc = length(find(Cls==ytest'))/length(ytest);
        %fprintf('DAE+%0.4f\n',acc);
        %
            Cls = cvKnn(P3_fs*zzztest{1}, P3_fs*zzztrain{1}, ytrain', 1);
            acc = length(find(Cls==ytest'))/length(ytest);
            %fprintf('DAE+%0.4f\n',acc);
        
        %     Cls = cvKnn(P4_fs*zzzztest{1}, P4_fs*zzzztrain{1}, ytrain', 1);
        %     acc = length(find(Cls==ytest'))/length(ytest);
        %     %fprintf('DAE+%0.4f\n',acc);
        rate_all_fs(1,loop) = acc;
        
            Cls = cvKnn([P_fs*ztest{1};P2_fs*zztest{1};P3_fs*zzztest{1}],...
                [P_fs*ztrain{1};P2_fs*zztrain{1};P3_fs*zzztrain{1}],...
                ytrain', 1);
            acc = length(find(Cls==ytest'))/length(ytest);
            %fprintf('DAE+%0.4f\n',acc);
            rate_all_fs2(1,loop) = acc;
        %%% 5. write on log file
        % fid = fopen(sprintf('%s/%s.txt',logpath,dataset),'a+');
        % fprintf(fid,'val err = %g, test err = %g\n', 100-acc_val, 100-acc_test);
        % fprintf(fid,'%s\n\n', fname);
        % fclose(fid);
        
    end
    %     rate_all
    %     mean(rate_all)
    rate_all_fs
    mean(rate_all_fs)
    rate_all_fs2
    mean(rate_all_fs2)
    std(rate_all_fs2)
end

