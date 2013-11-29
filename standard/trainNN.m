function model = trainNN(X,Y)

% Split in training and validation
n = size(X,1);
r = randperm(n);
ntr = round(n*0.75);
Xtrain = X(r(1:ntr),:);
Ytrain = Y(r(1:ntr),:);
Xvalid = X(r(ntr+1:end),:);
Yvalid = Y(r(ntr+1:end),:);

% Recode labels
Y_nn  = recodeLabels(Y(:));
Ytrain_nn  = recodeLabels(Ytrain(:));
Yvalid_nn  = recodeLabels(Yvalid(:));
% Recode inputs for validation
VV.P = Xvalid';
VV.T = Yvalid_nn';
% Parameters
[nin  nsam] = size(Xtrain');
[nout nsam] = size(Ytrain_nn');
limits = [min(Xtrain); max(Xtrain)]';
i=0;
for nh=2:30
    i=i+1;
    % Create network
    net = newff(limits,[nh nout],{'tansig' 'tansig'},'trainlm');
    net.trainParam.showWindow = false;
    method = 'trainlm'; % Levenberg-Marquardt algorithm for training
    net.trainParam.epochs = 100;       % Epochs or iterations
    % Train
    [net,tr,OUTPUTS,ERRORS,Pf,Af] = train(net,Xtrain',Ytrain_nn',[],[],VV,[]);
    % Predict
    [Ypred_NN,Pf,Af,EE,perf] = sim(net,Xvalid');
    [val Ypred_NN] = max(Ypred_NN);
    Ypred_NN = Ypred_NN';
    RES_NN    = assessment(Yvalid, Ypred_NN,'class');
    res(i,:) = [nh RES_NN.Kappa];
end

[val idx] = max(res(:,2));
nh = res(idx,1);
limits = [min(X); max(X)]';
net = newff(limits,[nh nout],{'tansig' 'tansig'},'trainlm');
net.trainParam.showWindow = false;
method = 'trainlm'; % Levenberg-Marquardt algorithm for training
net.trainParam.epochs = 10;       % Epochs or iterations
VV.P = X';
VV.T = Y_nn';
[model,tr,OUTPUTS,ERRORS,Pf,Af] = train(net,X',Y_nn',[],[],VV,[]);
