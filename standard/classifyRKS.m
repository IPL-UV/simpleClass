function Ypred = classifyRKS(Xtest,Xtrain,Ytrain)

Xtrain = Xtrain';
Xtest  = Xtest';
Ytrain = recodeLabels(Ytrain);

[d ntrain] = size(Xtrain)
D      = 2000; % number of features: magic number
w      = randn(D,d); 
Z      = cos(w*Xtrain);
Ztest  = cos(w*Xtest);

Kapprox = Z*Z';
Yapprox = Z*Ytrain;
lambda = max(Kapprox(:))*1e-3;  % regularization parameter: magic number!
alpha = (Kapprox + lambda*eye(D) )\Yapprox;
ntest=size(Xtest,2);
Yp = real(alpha'*Ztest)';
[kk Ypred] = max(Yp,[],2);
