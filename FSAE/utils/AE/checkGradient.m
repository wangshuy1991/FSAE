% check gradient of the the cost function
clear;
m = 3;   % number of examples
s0 = 7; % input dimension
s1 = 8; % number of hidden units layer 1
s2 = 9; % number of hidden units layer 2
s3 = 10; %
d = s1*s0 + s1 + s1*s2 + s2 + s3*s2 + s3 + s2 + s1 + s0; % this
layersizes = [s1, s2, s3];
theta = randn(d, 1);
theta(2) = 0;
data = randn(s0, m);

[cost,grad] = deepAutoencoder(theta, layersizes, data);
eps = 1e-5;

for i=1:d
    thetanew1 = theta;
    thetanew1(i) = theta(i) + eps;
    cost1 = deepAutoencoder(thetanew1, layersizes, data);

    thetanew2 = theta;
    thetanew2(i) = theta(i) - eps;
    cost2 = deepAutoencoder(thetanew2, layersizes, data);

    numgrad = (cost1 - cost2) / (2*eps);
    diff = abs(grad(i) - numgrad);
    assert(diff < eps, 'numerical check failed\n');
end
fprintf('cool, numerical check passed\n');