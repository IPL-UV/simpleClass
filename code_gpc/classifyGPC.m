function [Ypredtest probs mu s2] = classifyGPC(Xtest,Xtrain,Ytrain)

% GP for classification with Laplace method

numClasses = numel(unique(Ytrain));
ntrain = size(Xtrain,1);
ntest = size(Xtest,1);
% faster test: Predict in batches for large test matrices
if ntest<1000
    folds=1;
else
    folds = round(ntest/1000); % number of approximate folds for testing (a fold will contain roughly 1000 samples)
end
indices = crossvalind('Kfold',1:ntest,folds); % generate random indices to sample folds

% A classifier per class
for c = 1:numClasses
    y = - ones(ntrain,1);
    y(Ytrain == c,:) = +1;
    
    K = {'covSum', {'covSEard','covNoise'}};
    lengthscales = log((max(Xtrain)-min(Xtrain))/2)';
    SignalPower  = mean(var(y));
    NoisePower   = SignalPower/4;
    loghyper     = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower)];
    
    % optimize model
    [loghyper logmarglik] = minimize(loghyper, 'binaryLaplaceGP', -100, K, 'cumGauss', Xtrain, y);
    
    % test
    %[probs(:,c) mu(:,c) s2(:,c)] = binaryLaplaceGP(loghyper, K, 'cumGauss', Xtrain, y, Xtest);
          
    for f = 1:folds
        test = find(indices==f);  % select samples belonging to fold "f"
        [probs(test,c) mu(test,c) s2(test,c)] = binaryLaplaceGP(loghyper, K, 'cumGauss', Xtrain, y, Xtest(test,:));
    end
    
end

[~,Ypredtest] = max(probs,[],2);
