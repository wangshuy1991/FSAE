function [cost,grad] = deepAutoencoder(theta, layersizes, data)
% cost and gradient of a deep autoencoder 
% layersizes is a vector of sizes of hidden layers, e.g., 
% layersizes[2] is the size of layer 2
% this does not count the visible layer
% data is the input data, each column is an example
% the activation function of the last layer is linear, the activation
% function of intermediate layers is the hyperbolic tangent function

% WARNING: the code is optimized for ease of implemtation and
% understanding, not speed nor space

%% FORCING THETA TO BE IN MATRIX FORMAT FOR EASE OF UNDERSTANDING
% Note that this is not optimized for space, one can just retrieve W and b
% on the fly during forward prop and backprop. But i do it here so that the
% readers can understand what's going on

%%%%%%%%%%%%%%%%%%%%%%%%%wang.shuyang%%%%%%%%%%%%%%%%%%%
s0 = size(data,1);
s0(2:size(layersizes,2)+1) = layersizes;
layersizes = s0;
%layersizes = [size(data,1) layersizes];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
[h{i+1} dh{i+1}] = linearAct(bsxfun(@plus, W{i+1}*h{i}, b{i+1}));

%% COMPUTE COST
diff = h{i+1} - data; 
M = size(data,2); 
cost = 1/M * 0.5 * sum(diff(:).^2);
%%%%%%%%%%%%%%%%%%%%%%%% wang.shuyang %%%%%%%%%%%%%%%%%%%%%%
diff = diff.* dh{i+1}; 
%%%%%%%%%%%%%%%%%%%%%%%% wang.shuyang %%%%%%%%%%%%%%%%%%%%%%

%% BACKPROP
if nargout > 1
    outderv = 1/M * diff;    
    for i=2*(l-1):-1:2
        Wgrad{i} = outderv * h{i-1}';
        bgrad{i} = sum(outderv,2);        
        outderv = (W{i}' * outderv) .* dh{i-1};        
    end
    Wgrad{1} = outderv * data';
    bgrad{1} = sum(outderv,2);
        
    % handle tied-weight stuff        
    j = 1;
    for i=l:2*(l-1)
        Wgrad{l-j} = Wgrad{l-j} + Wgrad{i}';
        j = j + 1;
    end
    % dump the results to the grad vector
    grad = zeros(size(theta));
    lnew = 0;
    for i=1:l-1
        lold = lnew + 1;
        lnew = lnew + layersizes(i) * layersizes(i+1);
        grad(lold:lnew) = Wgrad{i}(:);
        lold = lnew + 1;
        lnew = lnew + layersizes(i+1);
        grad(lold:lnew) = bgrad{i}(:);
    end
    j = 1;
    for i=l:2*(l-1)
        lold = lnew + 1;
        lnew = lnew + layersizes(l-j);
        grad(lold:lnew) = bgrad{i}(:);
        j = j + 1;
    end
end 
end

%% USEFUL ACTIVATION FUNCTIONS
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