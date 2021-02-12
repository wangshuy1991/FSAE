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
for loop = 1:10
    load(strcat('../data/Coil/10Train/',num2str(loop),'.mat'))
    trainIdx = trainIdx(1:end/5*2,:);
    testIdx = testIdx(1:end/5*2,:);
    TestIdx = [TestIdx testIdx];
    TrainIdx = [TrainIdx trainIdx];
end

rate_all = zeros(1,10);
rate_all_fs = zeros(1,10);
options.Method = 'lbfgs';
options.maxIter = 20;
options.display = 'off';
options.TolX = 1e-7;

for lambda1 = [0.00001 0.0001 0.001 0.01 0.1]
    for lambda2 = [0.01 0.1 1 10 100 1000]
        parfor loop = 1:10
            xtrain = fea(TrainIdx(:,loop),:)';
            xtest = fea(TestIdx(:,loop),:)';
            
            ytrain = gnd(TrainIdx(:,loop));
            ytest = gnd(TestIdx(:,loop));
            
            
            %         [eigvec, eigval, ~, sampleMean] = PCA(xtrain',500);
            %         Wdims = size(eigvec, 2);
            %         xtrain = (bsxfun(@minus, xtrain', sampleMean) * eigvec(:, 1:Wdims))';
            % %     xtrain = [xtrain*diag(sparse(1./sqrt(sum(xtrain.^2))))];
            %     xtest = (bsxfun(@minus, xtest', sampleMean) * eigvec(:, 1:Wdims))';
            % %     xtest = [xtest*diag(sparse(1./sqrt(sum(xtest.^2))))];
            %     xtraintest = mapminmax([xtrain,xtest],-1,1);
            %     xtrain = xtraintest(:,1:size(ytrain));
            %     xtest = xtraintest(:,end-size(ytest)+1:end);
            
            %%% 1. load data
            %[xtrain, ytrain, xval, yval] = load_mnist(dataset);
            finalObjective = 70/500;
            layersizes = [1024];
            traindata = xtrain;
            trainlabel = ytrain;
            theta=[];
            rand('seed',1)
            theta = initializeWeights(traindata, layersizes);
            
            perm = randperm(size(traindata,2));
            traindata = traindata(:,perm);
            trainlabel = trainlabel(perm);
            batchSize = 200;%size(xtrain,2);
            %batchSize = size(xtrain,2);
            numiter = min(ceil(size(traindata,2)/batchSize), 1000);
            maxIter = 20;
            
            for bati = 1:numiter
                startIndex = mod((bati-1) * batchSize, size(traindata,2)) + 1;
                fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
                data = traindata(:, startIndex:startIndex + batchSize-1);
                [theta, obj] = minFunc( @deepAutoencoder, theta, options, layersizes, ...
                    data);
            end
            
            for i=1:maxIter
                ztrain = extractFeatures(theta, layersizes, xtrain );
                
                [Sb, Sw, L_b, L_w] = calculate_L(ztrain{1}',ytrain);
                feature_num = size(Sb,1)*0.9;
                [fList, feature_score, subset_score] = GraphScore(Sb, Sw, feature_num);
                fsnum = floor(feature_num*0.7);
                P_fs = zeros(fsnum,size(Sb,1));
                for j = 1:fsnum
                    P_fs(j,fList(j))=1;
                end
                
                for bati = 1:numiter
                    startIndex = mod((bati-1) * batchSize, size(traindata,2)) + 1;
                    fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
                    data = traindata(:, startIndex:startIndex + batchSize-1);
                    label = trainlabel(startIndex:startIndex + batchSize-1);
                    [~,~, L_b, L_w] = calculate_L(data',label);
                    [theta, obj] = minFunc( @deepAutoencoder_FS, theta, options, layersizes, ...
                        data,label,P_fs, L_b, L_w,lambda1,lambda2);
                    %obj
                end
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
            
            %%% 2. pretrain RBM
            % hyperpars_rbm;
            % params = rbm_set_params(dataset,numhid,epsilon,l2reg,pbias,plambda,kcd,maxiter,batchsize,savepath);
            % params.numvis = size(xtrain,1);
            % w_rbm = rbm_train(xtrain, params, usejacket);
            
            
            %%% 3. train PGBM
            % hyperpars_pgbm;
            % params = pgbm_set_params(dataset,numhid1,numhid2,epsilon,l2reg,pbias,plambda,kcd,ngibbs,use_meanfield,maxiter,batchsize,savepath);
            % params.numvis = size(xtrain,1);
            % fname = sprintf('pgbm_%s_vis%d_hid1_%02d_hid2_%02d_eps%g_l2reg%g_pb%g_pl%g_kcd%d_ngibbs%d_usemf%d_iter%d', ...
            %     params.dataset, params.numvis, params.numhid1, params.numhid2, params.epsilon, params.l2reg, params.pbias, params.plambda, params.kcd, params.ngibbs, params.use_meanfield, params.maxiter);
            % [w_pgbm, params] = pgbm_train(xtrain, params, w_rbm, ytrain, xval, yval, usejacket);
            
            
            %%% 4. test
            
            xval = xtest;
            yval = ytest;
            %[~, ~, ~, ~, xtest, ytest] = load_mnist(dataset);
            % ztrain = pgbm_inference(xtrain, w_pgbm, params);
            % zval = pgbm_inference(xval, w_pgbm, params);
            % ztest = pgbm_inference(xtest, w_pgbm, params);
            
            ztrain = extractFeatures(theta, layersizes, xtrain );
            %zval = extractFeatures(theta, layersizes, xval );
            ztest = extractFeatures(theta, layersizes, xtest );
            zval = ztest;
            %     out = fsTtest_multiclass_wrapper(ztrain', ytrain);
            %     [~,sidx] = sort(out.meanscores, 'descend');
            %     ztrain = ztrain(sidx(1:500),:);
            %     zval = zval(sidx(1:500),:);
            %     ztest = ztest(sidx(1:500),:);
            
            % [acc_train, acc_val, acc_test, bestC] = liblinear_wrapper([], xtrain, ytrain, xval, yval, xtest, ytest);
            % [acc_train, acc_val, acc_test, bestC]
            % [acc_train, acc_val, acc_test, bestC] = liblinear_wrapper([], ztrain{1}, ytrain, zval{1}, yval, ztest{1}, ytest);
            % [acc_train, acc_val, acc_test, bestC]
            % [acc_train, acc_val, acc_test, bestC] = liblinear_wrapper([], P_fs*ztrain{1}, ytrain, P_fs*zval{1}, yval, P_fs*ztest{1}, ytest);
            % [acc_train, acc_val, acc_test]
            
            Cls = cvKnn(xtest, xtrain, ytrain', 1);
            acc = length(find(Cls==ytest'))/length(ytest);
            
            fprintf('DAE+%0.4f\n',acc);
            
            Cls = cvKnn(ztest{1}, ztrain{1}, ytrain', 1);
            acc = length(find(Cls==ytest'))/length(ytest);
            
            fprintf('DAE+%0.4f\n',acc);
            rate_all(1,loop) = acc;
            
            Cls = cvKnn(P_fs*ztest{1}, P_fs*ztrain{1}, ytrain', 1);
            acc = length(find(Cls==ytest'))/length(ytest);
            
            fprintf('DAE+%0.4f\n',acc);
            rate_all_fs(1,loop) = acc;
            %%% 5. write on log file
            % fid = fopen(sprintf('%s/%s.txt',logpath,dataset),'a+');
            % fprintf(fid,'val err = %g, test err = %g\n', 100-acc_val, 100-acc_test);
            % fprintf(fid,'%s\n\n', fname);
            % fclose(fid);
            
        end
        rate_all
        mean(rate_all)
        rate_all_fs
        mean(rate_all_fs)
        lambda1
        lambda2
    end
end
