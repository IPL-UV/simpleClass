
clear;clc;close all;

fontname = 'Bookman';
fontsize = 14;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',3,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 0]);

warning off
randn('seed',1234)
rand('seed',1234)

format bank

addpath('./code_svm')
addpath('./code_gpc')
addpath('./standard')

%% Load image to do pixel-wise classification
load('IndianPines.mat')

% Sizes: rows x columns x bands
[r c b] = size(Xtotal);

% Number of classes
NumClases = max(unique(Ytotal))

% 1) Image to Matrix
%----------------------------------------------------------------------
XXtotal = reshape(Xtotal,r*c,b);
YYtotal = reshape(Ytotal,r*c,1); % YYtotal = Ytotal(:);
YYtotal = double(YYtotal);

% 1b) Reduce data dimensionality with PCA to 10 PCs
Nf = 10;
[V D] = eigs(cov(XXtotal),b);
XXtotal = XXtotal*V(:,1:Nf);

% 2) Scale data
%----------------------------------------------------------------------
XXtotal = scale(XXtotal);

% 3) Select a number of labeled pixels per class for training. (other
% smarter selections are possible, e.g. spatially, active learning,...)
%    Disregard background (label=0)
%----------------------------------------------------------------------

rate = 0.0001;       % Rate of selected pixels per class
Xtrain = [];
Ytrain = [];
for class=1:NumClases		% skip class 0
    ic = find(YYtotal==class);
    Npc = max(round(length(ic)*rate), Nf+1); % select the rate or at least as many samples/class as dimensions (for QDA/MAHAL)
    Xtrain = [Xtrain;
        XXtotal(ic(1:Npc),:)];
    Ytrain = [Ytrain; YYtotal(ic(1:Npc),:)];
end

% 4) Train and test with several standard classifiers
%----------------------------------------------------------------------

clc

% Foreground pixels
nozero = find(YYtotal~=0);

% Methods to be compared
% METHODS = {'LDA' 'QDA' 'MAHAL' 'KNN' 'TREE' 'BAG' 'BOOST' 'RF' 'NN' 'SVM' 'GPC'}
METHODS = {'LDA', 'SVM', 'GPC'}

MM=0;

if sum(strcmpi(METHODS,'LDA'))
    % (1) LDA
    MM=MM+1;
    disp('Training LDA ...')
    t = cputime;
    Ypred  = classify(XXtotal,Xtrain,Ytrain);
    Yp(:,MM)=Ypred;
    time(MM)   = cputime-t;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM)  = RES.Kappa;
end

if sum(strcmpi(METHODS,'QDA'))
    % (2) QDA
    MM=MM+1;
    disp('Training QDA ...')
    t = cputime;
    Ypred  = classify(XXtotal,Xtrain,Ytrain,'quadratic');
    Yp(:,MM)=Ypred;
    time(MM) = cputime-t;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'MAHAL'))
    % (3) MAHALANOBIS
    MM=MM+1;
    disp('Training Mahalanobis LDA ...')
    t = cputime;
    Ypred  = classify(XXtotal,Xtrain,Ytrain,'mahalanobis');
    Yp(:,MM)=Ypred;
    time(MM) = cputime-t;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'KNN'))
    % (4) k-NN
    MM=MM+1;
    disp('Training k-NN ...')
    t = cputime;
    Ypred  = knnclassify(XXtotal,Xtrain,Ytrain);
    Yp(:,MM)=Ypred;
    time(MM) = cputime-t;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'TREE'))
    % (5) TREES
    MM=MM+1;
    fprintf('Training Trees ... \n')
    t = cputime;
    model = treefit(Xtrain,Ytrain,'method','classification');
    time(MM) = cputime-t;
    Ypred = treeval(model,XXtotal);
    Yp(:,MM)=Ypred;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'BAG'))
    % (6) Bagging Trees
    MM=MM+1;
    fprintf('Training Bagging Trees ... \n')
    ntrees = 200; % Number of trees in the bag
    t = cputime;
    bag = fitensemble(Xtrain,Ytrain,'Bag',ntrees,'Tree','type','classification');
    time(MM) = cputime-t;
    Ypred = predict(bag,XXtotal);
    Yp(:,MM)=Ypred;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'BOOST'))
    % (7) AdaBoostM2 trees
    MM=MM+1;
    fprintf('Training Boosting Trees ... \n')
    ntrees = 200; % Number of trees for boosting
    t = cputime;
    boost = fitensemble(Xtrain,Ytrain,'AdaBoostM2',ntrees,'Tree','type','classification');
    time(MM) = cputime-t;
    Ypred = predict(boost,XXtotal);
    Yp(:,MM)=Ypred;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'RF'))
    % (8) Random forest
    MM=MM+1;
    ntrees = 200; % Number of trees for the forest
    t = cputime;
    forest = TreeBagger(ntrees,Xtrain,Ytrain);
    time(MM) = cputime-t;
    Ypred = predict(forest,XXtotal);
    Ypred = str2double(Ypred);
    Yp(:,MM)=Ypred;
    RES    = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'NN'))
    % (9) Neural networks
    MM=MM+1;
    disp('Training NNs ...')
    t = cputime;
    model = trainNN(Xtrain,Ytrain);
    time(MM) = cputime-t;
    Ypred = testNN(model,XXtotal);
    Yp(:,MM)=Ypred;
    RES    = assessment(YYtotal(nozero), Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'SVM'))
    % (10) Support vector machines (SVM)
    MM=MM+1;
    disp('Training SVM ...')
    t = cputime;
    Ypred = classifySVM(XXtotal,Xtrain,Ytrain);
    Yp(:,MM)=Ypred;
    time(MM) = cputime-t;
    RES = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

if sum(strcmpi(METHODS,'GPC'))
    % (11) Gaussian Process Classification (GPC)
    MM=MM+1;
    disp('Training GPC ...')
    t = cputime;
    [Ypred PROBS MUS S2] = classifyGPC(XXtotal,Xtrain,Ytrain);
    time(MM) = cputime-t;
    Yp(:,MM)=Ypred;
    RES = assessment(YYtotal(nozero),Ypred(nozero),'class');
    OAS(MM)    = RES.OA;
    KAPPA(MM) = RES.Kappa;
end

%----------------------------------------------------------------------
% 5) Results
%----------------------------------------------------------------------

% Accuracy

figure,
bar(OAS)
set(gca,'Xtick',1:11,'XTickLabel',METHODS)
ylabel('Overall accuracy, OA[%]')
grid

% Kappa statistic

figure,
bar(KAPPA)
set(gca,'Xtick',1:11,'XTickLabel',METHODS)
ylabel('\kappa statistic')
grid

% CPU times
figure,
barh(log10(time))
xlabel('')
set(gca,'YTickLabel',METHODS);
xlabel('CPU Time log([s])')
grid

% Sizes
[r c b] = size(Xtotal);

% RGB:
I = Xtotal(:,:,[30 20 10]);
RGB = reshape(I/max(I(:)),[r,c,3]);
figure,imagesc(RGB), axis off square, title('Ground truth')
% Ground truth
figure,imagesc(reshape(YYtotal,r,c)), axis off square, title('Ground truth')

% Classification maps
for mm=1:MM
    figure,imagesc(reshape(Yp(:,mm),r,c)), 
    axis off square, title([METHODS{mm} ', Kappa=' num2str(KAPPA(mm))])
end

% Sort figures in the screen
tile


