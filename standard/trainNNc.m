function [model,tr] = trainNN(X,Y)

% Split in training and validation
n = size(X,1);
% r = randperm(n);
% ntr = round(n*0.75);
% Xtrain = X(r(1:ntr),:);
% Ytrain = Y(r(1:ntr),:);
% Xvalid = X(r(ntr+1:end),:);
% Yvalid = Y(r(ntr+1:end),:);

% Define training, validation and test sets
[trainInd, valInd, testInd] = dividerand(n, 0.7, 0.15, 0.15);

% Recode labels
X = X(:,:);
Y_nn = ind2vec(Y(:)'); % recodeLabels(Y(:));
% Ytrain_nn = ind2vec(Ytrain(:)'); % recodeLabels(Ytrain(:));
% Yvalid_nn = ind2vec(Yvalid(:)'); % recodeLabels(Yvalid(:));

% Recode inputs for validation
% For new NN toolboxes, the way to do this is to use X and Y_nn directly
% with 'train'. It automatically separates data into train, val and test.
% VV.P = Xvalid';
% VV.T = Yvalid_nn';

% Parameters
% nin = size(Xtrain',1);
% nout = size(Ytrain_nn',1);
% limits = [min(Xtrain) ; max(Xtrain)]';
nhs = 10:10:30; % 10:5:30;
nns = 10; % number of NNs to train and select from
i = 0;
% method = 'trainlm'; % Levenberg-Marquardt algorithm for training
method = 'trainbr'; % Levenberg-Marquardt with Bayesian regularization
% method = 'trainscg'; % Scaled conjugate gradient backpropagation (default at least since R2011b)

net = cell(length(nhs),nns);
tr = cell(length(nhs),nns);
res = zeros(length(nhs),nns);

for nh = nhs
    i = i + 1;
    fprintf('  nh: %d\n', nh)
    
    % Create network (deprecated)
    %net = newff(limits, [nh nout], {'tansig' 'tansig'}, 'trainlm');
    
    for nn = 1:nns
        fprintf('        nn: %d\n', nn)
        net{i,nn} = patternnet(nh, method);
        net{i,nn}.trainParam.showWindow = false;
        net{i,nn}.trainParam.epochs = 100;
        
        net{i,nn}.divideFcn = 'divideind';
        net{i,nn}.divideParam.trainInd = trainInd;
        net{i,nn}.divideParam.valInd = valInd;
        net{i,nn}.divideParam.testInd = testInd;
        
        % Train
        %[net,tr,OUTPUTS,ERRORS,Pf,Af] = train(net, Xtrain', Ytrain_nn', [], [], VV, []);
        
        % Train using the whole set and the provided train/validation/test sets
        [net{i,nn},tr{i,nn}] = train(net{i,nn}, X', Y_nn, 'useParallel', 'yes');
        
        if any(tr{i,nn}.testInd ~= testInd)
            error('Some test index changed')
        end
        
        % Predict
        %[Ypred_NN,Pf,Af,EE,perf] = sim(net,Xvalid');
        %[val Ypred_NN] = max(Ypred_NN);
        %Ypred_NN = Ypred_NN';
        %RES_NN    = assessment(Yvalid, Ypred_NN, 'class');
        %res(i,:) = [nh RES_NN.Kappa];
        
        Ypred_NN = net{i,nn}(X(testInd,:)');
        Ypred_NN = vec2ind(Ypred_NN);
        RES_NN   = assessment(Y(testInd,:), Ypred_NN', 'class');
        res(i,nn) = RES_NN.Kappa;
        %restrn(i,:) = trn.best_vperf;
    end
    %[~,idx] = max(resnn);
end

[~,idx_nn] = max(max(res,[],1));
[~,idx_nh] = max(max(res,[],2));

model = net{idx_nh,idx_nn};
tr = tr{idx_nh,idx_nn};

% for debugging
save nn_data

% Return here with the best obtained model
return

% This is to re-train the network using all passed data.
% To use it no test data must be used (so not to loose 15% of training data)
% It may be achieved like (see foor loop, NOT YET TESTED!)

% nh = res(idx,1);
% limits = [min(X); max(X)]';
% net = newff(limits,[nh nout],{'tansig' 'tansig'},'trainlm');
% net = feedforwardnet(nh);
% net = configure(net, X', Y_nn');
% net.layers{end}.transferFcn = 'tansig';
tr = cell(1,nns);
for nn = 1:nns
    fprintf('    nn: %d\n', nn)
    net{nn} = patternnet(nh, method);
    net{nn}.trainParam.showWindow = false;
    net{nn}.trainParam.epochs = 100; % Number of epochs
    
    net{nn}.divideFcn = 'divideind';
    net{i,nn}.divideParam.trainInd = trainInd;
    net{i,nn}.divideParam.valInd = [ valInd testInd ]; % joint val. and test sets
    net{i,nn}.divideParam.testInd = []; % use an empty validation test 
    
    % VV.P = X';
    % VV.T = Y_nn';
    % [model,tr,OUTPUTS,ERRORS,Pf,Af] = train(net,X',Y_nn',[],[],VV,[]);
    [net{nn},tr{nn}] = train(net{nn}, X', Y_nn);
    
    Ypred_NN = net{nn}(X(tr{nn}.valInd,:)');
    Ypred_NN = vec2ind(Ypred_NN);
    RES_NN   = assessment(Y(tr{nn}.valInd,:), Ypred_NN', 'class');
    resnn(nn) = RES_NN.Kappa;    
end
[~,idx] = max(resnn);
model = net{idx};
tr = tr{idx};
