function [theta,P_fs] = FSAE(traindata,trainlabel,layersizes,options,finalObjective,lambda1,lambda2,FSratio)

theta=[];
rand('seed',1)
theta = initializeWeights(traindata, layersizes);

perm = randperm(size(traindata,2));
traindata = traindata(:,perm);
trainlabel = trainlabel(perm);
%batchSize =510;%size(xtrain,2);
batchSize = size(traindata,2);
numiter = min(ceil(size(traindata,2)/batchSize), 1000);
maxIter = 20;

for bati = 1:numiter
    startIndex = mod((bati-1) * batchSize, size(traindata,2)) + 1;
    %fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
    data = traindata(:, startIndex:min(startIndex + batchSize-1,size(traindata,2)));
    [theta, obj] = minFunc( @deepAutoencoder, theta, options, layersizes, ...
        data);
end

for i=1:maxIter
    ztrain = extractFeatures(theta, layersizes, traindata);
    
    [Sb, Sw, L_b, L_w] = calculate_L(ztrain{1}',trainlabel);
    feature_num = size(Sb,1)*0.9;
    [fList, feature_score, subset_score] = GraphScore(Sb, Sw, feature_num);
    fsnum = floor(feature_num*FSratio);
    P_fs = zeros(fsnum,size(Sb,1));
    for j = 1:fsnum
        P_fs(j,fList(j))=1;
    end
    
    for bati = 1:numiter
        startIndex = mod((bati-1) * batchSize, size(traindata,2)) + 1;
        %fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
        data = traindata(:, startIndex:startIndex + batchSize-1);
        label = trainlabel(startIndex:startIndex + batchSize-1);
        [~,~, L_b, L_w] = calculate_L(data',label);
%         [theta, obj] = minFunc( @deepAutoencoder, theta, options, layersizes, ...
%         data);
        [theta, obj] = minFunc( @deepAutoencoder_FS, theta, options, layersizes, ...
            data,label,P_fs, L_b, L_w,lambda1,1/subset_score);
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
end

