clear
clc
load COIL100.mat
fea = double(fea)/255;
ratio=0.5;
d = size(fea,2);
len = floor(d*ratio);

corruption = rand(len,1);
start = randperm(d-len);
for i = 1:size(fea,1)
    start = randperm(d-len);
    corruption = rand(len,1);
    fea(i,start(20:20+len-1)) = corruption;
    %fea(:,i) = fea(:,i) / norm(fea(:,i));
end
fea = fea*255;