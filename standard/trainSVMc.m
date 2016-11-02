function [model,error,yp,is,ic,ie,groups] = trainSVMc(yt,xt,params)

% function [model,error,yp,is,ic,ie,groups] = trainSVMc(yt,xt,params)
%
% Inputs:
%   Y:          labels.
%   X:          samples with features in columns.
%   params:     structure with the following fields:
%     percent:  percentage of training samples. Default is 30%.
%     vfold:    if ~= 0 vfold is used for training, and vfold is the number
%               of folds to use. If == 0, a simple crossval using 'percent'
%               of samples for training set is used. Defauls is 5.
%     svm_type: classification or regression, the same as -s parameter of 
%               libsvm: 0: C-SVC, 1: nu-SVC, 2: one-class, 3: eps-SVR, 4:nu-SVR.
%     knl_type: -t parameter of libsvm: 0: linear, 1: polynomial, 2: rbf, 3: sigmoid, 4: precomputed.
%               At present, precomputed kernel is RBF kernel computed using 'kernelmatrix'.
%     sigma:    optional: vector of gamma values to test, default g = logspace(-2,2,20);
%     C:        optional: vector of C or nu values to test, default C = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3];
%     e:        optional: vector of epsilon values to test (for SVR), default e = [0.01:0.01:0.1];
%
% Outputs:
%   model:      obtained model.
%   error:      output of assessment.
%   yp:         classification / value.
%   is,ic,ie:   indexes of best free parameters found.
%   groups:     vfold calculated groups
%
% (c) jordi@uv.es, 2007-08

% Avoid problems using always double values
yt = double(yt);
xt = double(xt);

% OC-SVM
if params.svm_type == 2
    yt(yt == params.target) =  1; % Targets
    yt(yt ~= params.target) = -1; % Outliers
end

% Default values
if ~isfield(params,'percent')
    params.percent = 0.3;
end
if ~isfield(params,'vfold')
    params.vfold = 5;
end
if ~isfield(params,'sigma')
    params.sigma = logspace(-2,2,20);
end
if ~isfield(params,'C')
    params.C = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3];
end
if ~isfield(params,'pureoc')
    params.pureoc = 0;
end
if  ~isfield(params,'verb')
    params.verb = 1;
end
if params.svm_type == 3 || params.svm_type == 4 % SVR
    if ~isfield(params,'e')
        params.e = 0.01:0.01:0.1; %[0.01 0.1:0.1:0.5];
        ie = 2; % Para e, selecciona 0.1
    else
        [~,ie]= min(abs(params.e-0.1)); % o lo mas cercano a 0.1
    end
else
    params.e  = 0.1; % SVC does not use epsilon, fix its value
    ie = 1;
end

sigma = params.sigma;
C  = params.C;
e  = params.e;

% Selecciona lo mas proximo a C=10 y g=1
[~,ic] = min(abs(C-10));
[~,is] = min(abs(sigma-1));

% Pure OC
if params.pureoc
    % Just remove all outliers
    xt = xt(yt == params.target,:);
    yt = yt(yt == params.target);
end

% For assessment function
if params.svm_type < 3
    assespar = 'class';
else
    assespar = 'regress';
end

% Other parameters
verb = params.verb;
fine_tuning = 0;
vueltas = 2;

if verb, fprintf('  Adjusting free parameters ...\n'), end

for vuelta = 1:vueltas
    % Sigma fine tunning
    if fine_tuning && vuelta > 1
        sigma = logspace(log10(sigma(is)/10), log10(sigma(is)*10), length(sigma));
    end
    
    % Search sigma
    if verb, fprintf('  Sigma ...\n'), end
    params.sigma = sigma;
    params.C = C(ic);
    params.e = e(ie);
    [new_is,~,params] = testSVMParams(xt,yt,params);
    
    if vuelta > 1 && is == new_is
        if verb, fprintf('  Same sigma found, skipping C and eps\n'), end
        break
    end
    % else
    is = new_is;

    % Search C
    if verb, fprintf('  C ...\n'), end
    params.sigma = sigma(is);
    params.C = C;
    [ic,~,params] = testSVMParams(xt,yt,params);

    % Epsilon is for SVR only
    if params.svm_type == 3 || params.svm_type == 4
        if verb, fprintf('  Eps ...\n'), end
        params.C = C(ic);
        params.e = e;
        [ie,~,params] = testSVMParams(xt,yt,params);
    end
end

% Best parameters found, train a model with them

if verb, fprintf('  Free parameters: sigma %f, C %f and eps %f ...\n', sigma(is), C(ic), e(ie)), end

if params.knl_type == 4
    libpar = sprintf('-s %d -t 4 -c %f -p %f', params.svm_type, C(ic), e(ie));
    kt = kernelmatrix('rbf',xt',xt',sigma(is));
else
    libpar = sprintf('-s %d -t %d -g %f -c %f -p %f', ...
        params.svm_type, params.knl_type, 1/(2*sigma(is)*sigma(is)), C(ic), e(ie));
end

% nu-SVC, OC-SVM or nu-SVR
if params.svm_type == 1 || params.svm_type == 2 || params.svm_type == 4
    % Just replace -c per -n
    libpar = regexprep(libpar,'-c','-n');
end

% OC-SVM: use only target class for training
if params.svm_type == 2
    ct = find(yt == params.target);
else
    ct = 1:length(yt);
end

% Final model and test are obtained using all samples
if params.knl_type == 4
    model = svmtrain(yt(ct),kt(ct,ct),libpar);
    yp = svmpredict(yt,kt(:,ct),model);
else
    model = svmtrain(yt(ct),xt(ct,:),libpar);
    yp = svmpredict(yt,xt,model);
end
% s = warning('off','MATLAB:divideByZero');
error = assessment(yt,yp,assespar);
% warning(s);
if verb
    if params.svm_type < 3
        fprintf('  Kappa: %f', error.Kappa)
    else
        fprintf('  RMSE: %f', error.RMSE)
    end
    fprintf(', SV: %d (%4.1f%%)\n', model.totalSV, 100*model.totalSV/length(yt))
end

if isfield(params,'groups')
    groups = params.groups;
else
    groups = [];
end
