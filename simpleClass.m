
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

rate = 0.05;       % Rate of selected pixels per class
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
% METHODS = {'LDA' 'QDA' 'MAHALANOBIS' 'k-NN' 'TREE' 'BAG' 'ADABOOST' 'RF' 'NN' 'SVM' 'GPC'}
METHODS = {'LDA' 'SVM' 'GPC'}

% (1) LDA
disp('Training LDA ...')
t = cputime;
Ypred_LDA  = classify(XXtotal,Xtrain,Ytrain);
time_LDA = cputime-t;
RES_LDA    = assessment(YYtotal(nozero),Ypred_LDA(nozero),'class');
OAS_LDA    = RES_LDA.OA;
KAPPA_LDA = RES_LDA.Kappa

% % (2) QDA
% disp('Training QDA ...')
% t = cputime;
% Ypred_QDA  = classify(XXtotal,Xtrain,Ytrain,'quadratic');
% time_QDA= cputime-t;
% RES_QDA    = assessment(YYtotal(nozero),Ypred_QDA(nozero),'class');
% OAS_QDA    = RES_QDA.OA;
% KAPPA_QDA = RES_QDA.Kappa
% 
% % (3) MAHALANOBIS
% disp('Training Mahalanobis LDA ...')
% t = cputime;
% Ypred_MAHALANOBIS  = classify(XXtotal,Xtrain,Ytrain,'mahalanobis');
% time_MAHAL = cputime-t;
% RES_MAHALANOBIS    = assessment(YYtotal(nozero),Ypred_MAHALANOBIS(nozero),'class');
% OAS_MAHALANOBIS    = RES_MAHALANOBIS.OA;
% KAPPA_MAHALANOBIS = RES_MAHALANOBIS.Kappa
% 
% % (4) k-NN
% disp('Training k-NN ...')
% t = cputime;
% Ypred_KNN  = knnclassify(XXtotal,Xtrain,Ytrain);
% time_KNN = cputime-t;
% RES_KNN    = assessment(YYtotal(nozero),Ypred_KNN(nozero),'class');
% OAS_KNN    = RES_KNN.OA;
% KAPPA_KNN = RES_KNN.Kappa
% 
% % (5) TREES
% fprintf('Training Trees ... \n')
% t = cputime;
% model_TREE = treefit(Xtrain,Ytrain,'method','classification');
% time_TREE = cputime-t;
% Ypred_TREE = treeval(model_TREE,XXtotal);
% RES_TREE    = assessment(YYtotal(nozero),Ypred_TREE(nozero),'class');
% OAS_TREE    = RES_TREE.OA;
% KAPPA_TREE = RES_TREE.Kappa
% 
% % (6) Bagging Trees
% fprintf('Training Bagging Trees ... \n')
% ntrees = 200; % Number of trees in the bag
% t = cputime;
% bag = fitensemble(Xtrain,Ytrain,'Bag',ntrees,'Tree','type','classification');
% time_BAG = cputime-t;
% Ypred_BAG = predict(bag,XXtotal);
% RES_BAG    = assessment(YYtotal(nozero),Ypred_BAG(nozero),'class');
% OAS_BAG    = RES_BAG.OA;
% KAPPA_BAG = RES_BAG.Kappa
% 
% % (7) AdaBoostM2 trees
% fprintf('Training Boosting Trees ... \n')
% ntrees = 200; % Number of trees for boosting
% t = cputime;
% boost = fitensemble(Xtrain,Ytrain,'AdaBoostM2',ntrees,'Tree','type','classification');
% time_BOOST = cputime-t;
% Ypred_BOOST = predict(boost,XXtotal);
% RES_BOOST    = assessment(YYtotal(nozero),Ypred_BOOST(nozero),'class');
% OAS_BOOST    = RES_BOOST.OA;
% KAPPA_BOOST = RES_BOOST.Kappa
% 
% % (8) Random forest
% ntrees = 200; % Number of trees for the forest
% t = cputime;
% forest = TreeBagger(ntrees,Xtrain,Ytrain);
% time_RF = cputime-t;
% Ypred_RF = predict(forest,XXtotal);
% Ypred_RF = str2double(Ypred_RF);
% RES_RF    = assessment(YYtotal(nozero),Ypred_RF(nozero),'class');
% OAS_RF    = RES_RF.OA;
% KAPPA_RF = RES_RF.Kappa
% 
% % (9) Neural networks
% disp('Training NNs ...')
% t = cputime;
% model = trainNN(Xtrain,Ytrain);
% time_NN = cputime-t;
% Ypred_NN = testNN(model,XXtotal);
% RES_NN    = assessment(YYtotal(nozero), Ypred_NN(nozero),'class');
% OAS_NN    = RES_NN.OA;
% KAPPA_NN = RES_NN.Kappa

% (10) Support vector machines (SVM)
disp('Training SVM ...')
t = cputime;
Ypred_SVM = classifySVM(XXtotal,Xtrain,Ytrain);
time_SVM = cputime-t;
RES_SVM = assessment(YYtotal(nozero),Ypred_SVM(nozero),'class');
OAS_SVM    = RES_SVM.OA;
KAPPA_SVM = RES_SVM.Kappa

% (11) Gaussian Process Classification (GPC)
disp('Training GPC ...')
t = cputime;
[Ypred_GPC PROBS MUS S2] = classifyGPC(XXtotal,Xtrain,Ytrain);
time_GPC = cputime-t;
RES_GPC = assessment(YYtotal(nozero),Ypred_GPC(nozero),'class');
OAS_GPC    = RES_GPC.OA;
KAPPA_GPC = RES_GPC.Kappa


break
%----------------------------------------------------------------------
% 5) Results
%----------------------------------------------------------------------

% Accuracy

figure,
bar([OAS_LDA,OAS_QDA,OAS_MAHALANOBIS,OAS_KNN,OAS_TREE,OAS_BAG,OAS_BOOST,OAS_RF,OAS_NN,OAS_SVM,OAS_GPC])
set(gca,'Xtick',1:11,'XTickLabel',METHODS)
ylabel('Overall accuracy, OA[%]')
grid

% Kappa statistic

figure,
bar([KAPPA_LDA,KAPPA_QDA,KAPPA_MAHALANOBIS,KAPPA_KNN,KAPPA_TREE,KAPPA_BAG,KAPPA_BOOST,KAPPA_RF,KAPPA_NN,KAPPA_SVM,KAPPA_GPC])
set(gca,'Xtick',1:11,'XTickLabel',METHODS)
ylabel('\kappa statistic')
grid

% CPU times
figure,
barh(log10([time_LDA,time_QDA,time_MAHAL,time_KNN,time_TREE,time_BAG,time_BOOST,time_RF,time_NN,time_SVM,time_GPC]))
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
figure,imagesc(reshape(Ypred_LDA,r,c)), axis off square, title(['LDA, Kappa=' num2str(KAPPA_LDA)])
figure,imagesc(reshape(Ypred_QDA,r,c)), axis off square, title(['QDA, Kappa=' num2str(KAPPA_QDA)])
figure,imagesc(reshape(Ypred_MAHALANOBIS,r,c)), axis off square, title(['MAHALANOBIS, Kappa=' num2str(KAPPA_MAHALANOBIS)])
figure,imagesc(reshape(Ypred_KNN,r,c)), axis off square, title(['k-NN, Kappa=' num2str(KAPPA_KNN)])
figure,imagesc(reshape(Ypred_TREE,r,c)), axis off square, title(['TREE, Kappa=' num2str(KAPPA_TREE)])
figure,imagesc(reshape(Ypred_BAG,r,c)), axis off square, title(['BAG, Kappa=' num2str(KAPPA_BAG)])
figure,imagesc(reshape(Ypred_BOOST,r,c)), axis off square, title(['BOOST, Kappa=' num2str(KAPPA_BOOST)])
figure,imagesc(reshape(Ypred_RF,r,c)), axis off square, title(['RF, Kappa=' num2str(KAPPA_RF)])
figure,imagesc(reshape(Ypred_NN,r,c)), axis off square, title(['NN, Kappa=' num2str(KAPPA_NN)])
figure,imagesc(reshape(Ypred_SVM,r,c)), axis off square, title(['SVM, Kappa=' num2str(KAPPA_SVM)])
figure,imagesc(reshape(Ypred_GPC,r,c)), axis off square, title(['SVM, Kappa=' num2str(KAPPA_GPC)])

% % Sort figures in the screen
tile
