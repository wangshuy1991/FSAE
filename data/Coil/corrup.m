clear
clc
load COIL100.mat
ratio=0.5;
d = size(fea,2);
len = floor(d*ratio);

corruption = rand(len,1);

for i = 1:size(fea,1)
   indx = randperm(d);
   fea(i,indx(1:len)) = corruption*255;
end
