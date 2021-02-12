datasetpath = 'D:\Facial Beauty\Attractiveness\Eleven_Fold\Downsampling\fold1';
load([datasetpath,'/', 'Bfeature_R.mat'])
Bfeature=zeros(size(h_B{1,end/2},2),size(h_B{1,end/2},1));
for i=1:size(Bfeature,1)
    Bfeature(i,:)=h_B{1,end/2}(:,i)'-h_NB{1,end/2}(:,i)';
end

datasetpath = 'D:\Facial Beauty\Attractiveness\Eleven_Fold\Downsampling\fold1';
load([datasetpath,'/', 'NBfeature.mat'])
NBfeature=zeros(size(h_B{1,end/2},2),size(h_B{1,end/2},1));
for i=1:size(NBfeature,1)
    NBfeature(i,:)=h_B{1,end/2}(:,i)'-h_NB{1,end/2}(:,i)';
end

datasetpath = 'D:\Facial Beauty\Attractiveness\Dataset\PCA';
load([datasetpath,'/', 'Beautydata.mat'])
Bfeature=zeros(size(traindata,2),size(traindata,1));
for i=1:size(Bfeature,1)
    Bfeature(i,:)=traindata(:,i)';
end
datasetpath = 'D:\Facial Beauty\Attractiveness\Dataset\PCA';
load([datasetpath,'/', 'NBeautydata.mat'])
NBfeature=zeros(size(traindata,2),size(traindata,1));
for i=1:size(NBfeature,1)
    NBfeature(i,:)=traindata(:,i)';
end

total_feature=[Bfeature' NBfeature']';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% normalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:size(total_feature,1)
    total_feature(i,:)= total_feature(i,:)/sum(total_feature(i,:));
end
total_label=zeros(size(Bfeature,1)+size(NBfeature,1),1);

total_label(1:size(Bfeature,1),:)=0;
total_label(size(Bfeature,1)+1:size(total_label,1),:)=1;

total_data=[total_label total_feature];

perm = randperm(size(total_data,1));
total_data = total_data(perm,:);

trainnum=1500;
testnum=size(total_data,1)-trainnum;

training_feature=total_data(1:trainnum,2:size(total_data,2));
training_label=total_data(1:trainnum,1);
testing_feature=total_data(trainnum+1:trainnum+testnum,2:size(total_data,2));
testing_label=total_data(trainnum+1:trainnum+testnum,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% E-distance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1
    datasetpath = ['D:\Facial Beauty\Attractiveness\Eleven_Fold\Downsampling\fold',int2str(i),'\'];
    load([datasetpath,'Bfeature_R.mat']);
    label1=label(1:end-2,:);
    load([datasetpath,'Bfeature_P.mat']);
    label2=label(1:end-2,:);
    load([datasetpath,'Bfeature_llr.mat']);
    label3=label(1:end-2,:);
    Bfeature=[label1 label2 label3];
    load([datasetpath,'NBfeature_R.mat']);
    label1=label(1:end-2,:);
    load([datasetpath,'NBfeature_P.mat']);
    label2=label(1:end-2,:);
    load([datasetpath,'NBfeature_llr.mat']);
    label3=label(1:end-2,:);
    NBfeature=[label1 label2 label3];
end
total_feature=[Bfeature' NBfeature']';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% normalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:size(total_feature,1)
    total_feature(i,:)= total_feature(i,:)/sum(total_feature(i,:));
end
total_label=zeros(size(Bfeature,1)+size(NBfeature,1),1);

total_label(1:size(Bfeature,1),:)=0;
total_label(size(Bfeature,1)+1:size(total_label,1),:)=1;

total_data=[total_label total_feature];

trainnum=1000;
testnum=size(total_data,1)/2-trainnum;

training_feature=[total_data(1:trainnum,2:end)' total_data(1+trainnum+testnum:2*trainnum+testnum,2:end)']';
training_label=[total_data(1:trainnum,1)' total_data(1+trainnum+testnum:2*trainnum+testnum,1)']';
testing_feature=[total_data(trainnum+1:trainnum+testnum,2:end)' total_data(end-testnum+1:end,2:end)']';
testing_label=[total_data(trainnum+1:trainnum+testnum,1)' total_data(end-testnum+1:end,1)']';




