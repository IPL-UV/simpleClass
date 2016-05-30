function [Ypredtest probs mu s2] = classifyGPC(Xtest,Xtrain,Ytrain)

% GP for classification with Laplace method

numClasses = numel(unique(Ytrain));
[ntrain,d] = size(Xtrain);

% A classifier per class
for c=1:numClasses
    y = - ones(ntrain,1);
    idx = find(Ytrain==c);
    y(idx,:) = +1;
    
    K = {'covSum', {'covSEard','covNoise'}};
    lengthscales = log((max(Xtrain)-min(Xtrain))/2)';
    SignalPower  = mean(var(y));
    NoisePower   = SignalPower/4;
    loghyper     = [lengthscales; 0.5*log(SignalPower); 0.5*log(NoisePower)];

    % optimize model
    [loghyper logmarglik] = minimize(loghyper, 'binaryLaplaceGP', -100, K, 'cumGauss', Xtrain, y);
    [probs(:,c) mu(:,c) s2(:,c)] = binaryLaplaceGP(loghyper, K, 'cumGauss', Xtrain, y, Xtest);
    
end

[val Ypredtest] = max(probs,[],2)

