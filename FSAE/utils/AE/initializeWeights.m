function [theta] = initializeWeights(traindata, layersizes)
%% Random initialization
% X. Glorot, Y. Bengio. 
% Understanding the difï¬?culty of training deep feedforward neural networks.
% AISTATS 2010.
% QVL: this initialization method appears to perform better than 
% theta = randn(d,1);

%%%%%%%%%%%%%%%%%%%%%%%%%wang.shuyang%%%%%%%%%%%%%%%%%%%
s0 = size(traindata,1);
s0(2:size(layersizes,2)+1) = layersizes;
layersizes = s0;
%layersizes = [s0 layersizes];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    r  = sqrt(6) / sqrt(layersizes(i+1)+layersizes(i));   
    A = rand(layersizes(i+1), layersizes(i))*2*r - r; %reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    theta(lold:lnew) = A(:);
    lold = lnew + 1;
    lnew = lnew + layersizes(i+1);
    A = zeros(layersizes(i+1),1);
    theta(lold:lnew) = A(:);
end
j = 1;
for i=l:2*(l-1)
    lold = lnew + 1;
    lnew = lnew + layersizes(l-j);
    theta(lold:lnew)= zeros(layersizes(l-j),1);
    j = j + 1;
end
theta = theta';
layersizes = layersizes(2:end);
end