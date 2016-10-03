function Ypred_SVM = classifySVM(XXtotal,Xtrain,Ytrain)

meanSigma = mean(pdist(Xtrain));
sigmaMin = log10(meanSigma*0.1);
sigmaMax = log10(meanSigma*10);
sigma = logspace(sigmaMin,sigmaMax,30);

resultsSVM = zeros(300,3);
j = 0;
for ss = sigma
    Ktrain = kernelmatrix('rbf',Xtrain',Xtrain',ss); % construyo el kernel
    for cc=logspace(-1,5,10)
        j = j + 1;
        model = svmtrain(Ytrain,Ktrain,['-t 4 -v 10 -c ' num2str(cc)]);
        resultsSVM(j,:) = [ss cc model];    % guarda los aciertos promedio
    end
end;

% Select the best combination of parameters
[~,j] = max(resultsSVM(:,3));
sigma = resultsSVM(j,1);
C     = resultsSVM(j,2);

% Train the final model
Ktrain = kernelmatrix('rbf',Xtrain',Xtrain',sigma);
model  = svmtrain(Ytrain,Ktrain,['-t 4 -c ' num2str(C)]);

% Predict in the whole image
Ktest = kernelmatrix('rbf',Xtrain',XXtotal',sigma);
YYtotal = zeros(size(XXtotal,1),1); 
Ypred_SVM = svmpredict(YYtotal,Ktest',model);
