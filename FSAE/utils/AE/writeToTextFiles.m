%% WRITE WEIGHTS TO FILES
% so that we can evaluate the validity of convergence
s0 = size(traindata,1);
layersizes = [s0 layersizes];
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W{i} = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    
    dlmwrite(['W', num2str(i), '.txt'], W{i});
    lold = lnew + 1;
    lnew = lnew + layersizes(i+1);
    b{i} = theta(lold:lnew);
    dlmwrite(['b', num2str(i), '.txt'], b{i});
end
% handle tied-weight stuff
j = 1;
for i=l:2*(l-1)
    lold = lnew + 1;
    lnew = lnew + layersizes(l-j);    
    b{i} = theta(lold:lnew);
    dlmwrite(['b', num2str(i), '.txt'], b{i});
    j = j + 1;
end
assert(lnew == length(theta), 'Error: dimension does not match\n')
