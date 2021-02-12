function h = extractFeatures(theta, layersizes, data)
% extract features from the trained autoencoder. After this,
% h{1}... h{length(layersizes)} can be used as features.
layersizes = [size(data,1) layersizes];
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W{i} = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    lold = lnew + 1;
    lnew = lnew + layersizes(i+1);
    b{i} = theta(lold:lnew);
end
% handle tied-weight stuff
j = 1;
for i=l:2*(l-1)
    lold = lnew + 1;
    lnew = lnew + layersizes(l-j);
    W{i} = W{l - j}';
    b{i} = theta(lold:lnew);
    j = j + 1;
end
assert(lnew == length(theta), 'Error: dimensions of theta and layersizes do not match\n')


%% FORWARD PROP
for i=1:2*(l-1)-1
    if i==1
        [h{i} dh{i}] = tanhAct(bsxfun(@plus, W{i}*data, b{i}));
    else
        [h{i} dh{i}] = tanhAct(bsxfun(@plus, W{i}*h{i-1}, b{i}));
    end
end
h{i+1} = linearAct(bsxfun(@plus, W{i+1}*h{i}, b{i+1}));

end

%% USEFUL ACTIVATION FUNCTIONS
%% COPIED FROM DEEPAUTOENCODER.M TO REDUCE THE NUMBER OF FILES
    function [a da] = sigmoidAct(x)
        
        a = 1 ./ (1 + exp(-x));
        if nargout > 1
            da = a .* (1-a);
        end
    end

    function [a da] = tanhAct(x)
        a = tanh(x);
        if nargout > 1
            da = (1-a) .* (1+a);
        end
    end

    function [a da] = linearAct(x)
        a = x;
        if nargout > 1
            da = ones(size(a));
        end
    end
