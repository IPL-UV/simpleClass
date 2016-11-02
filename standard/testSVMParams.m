function [ifreepar,error,params] = testSVMParams(xt,yt,params)

% function [ifreepar,error,params] = testSVMParams(xt,yt,params)
%
% This function tests a set of parameter values (param) using a train set
% (xt,yt) using vfold (vf) or a training/validation subset (ct,cv).
%
% Inputs:
%    xt,yt:       train signal.
%    params:      structure with the following fields
%      sigma,C,e: vector of G,C or Eps. values to try. Only one of these
%                 should have more than one element.
%      svm_type:  type of SVM (-s on libsvm).
%      knl_type:  kernel to use (-t on libsvm).
%      percent:   if no vfold, percetage of samples used as train set.
%      vfold:     a number indicating the number of folds for vfold, or
%                 zero to disable vfold. If > 0, ct and cv are not used.
%      groups:    vfold groups. If this field doesn't exist, a set of new
%                 groups are automatically calculated.
%      ct,cv:     train and test sets if vfold is not used. Automatically
%                 calculated is they don't exist.
%
% Outputs:
%    ifreepar:      index of best free parameter found.
%    error:         Kappa (SVC) or RMSE (SVR) of the best parameter.
%    params.groups: vfold groups.
%    params.ct,cv:  train and test sets according percent.
%
% (c) jordi@uv.es, 2007-08

if params.svm_type < 3
    assespar = 'class';
else
    assespar = 'regress';
end

verb = params.verb;

% Create groups for vfold or sets for crossval
if params.vfold
    if ~isfield(params,'groups')
        params.groups = folds(length(yt),params.vfold);
    end
else
    if ~isfield(params,'ct') || ~isfield(params,'cv')
        [params.ct,params.cv] =  getsets(yt, unique(yt), params.percent, 0, 'regress', 1);
    end
end

% To be set for precomputed kernels and testing sigma
par_is_sigma = 0;

switch params.knl_type
    case 4 % precomputed
        if numel(params.sigma) > 1
            libsvm = sprintf('-s %d -t 4 -c %f -p %f', params.svm_type, params.C(1), params.e(1));
            freepar = params.sigma;
            par_is_sigma = 1;
        elseif numel(params.C) > 1
            libsvm = sprintf('-s %d -t 4 -p %f -c %%f', params.svm_type, params.e(1));
            freepar  = params.C;
        else
            libsvm = sprintf('-s %d -t 4 -c %f -p %%f', params.svm_type, params.C(1));
            freepar  = params.e;
        end
    otherwise
        % Convert sigma to gamma
        params.g = 1./(2*params.sigma.*params.sigma);
        if numel(params.g) > 1
            libsvm  = sprintf('-s %d -t %d -c %f -p %f -g %%f', params.svm_type,params.knl_type, params.C(1), params.e(1));
            freepar = params.g;
        elseif numel(params.C) > 1
            libsvm  = sprintf('-s %d -t %d -g %f -p %f -c %%f', params.svm_type, params.knl_type, params.g(1), params.e(1));
            freepar = params.C;
        else
            libsvm  = sprintf('-s %d -t %d -g %f -c %f -p %%f', params.svm_type, params.knl_type, params.g(1), params.C(1));
            freepar = params.e;
        end
end

% nu-SVC, OC-SVM or nu-SVR
if params.svm_type == 1 || params.svm_type == 2 || params.svm_type == 4
    % Just replace -c per -n
    libsvm = regexprep(libsvm,'-c','-n');
end

% Reserve memory
if params.vfold
    error = zeros(params.vfold,length(freepar));
else
    error = zeros(1,length(freepar));
end

% If we are not testing sigma, kernel matrix is constant
if params.knl_type == 4 && ~par_is_sigma
    kt = kernelmatrix('rbf',xt',xt',params.sigma(1));
end

for i = 1:length(freepar)
    
    %if verb, fprintf('  %10.2f ... ', param(i)), end
    
    libpar = sprintf(libsvm, freepar(i));
    if par_is_sigma
        % Calculate kernel matrix
        kt = kernelmatrix('rbf', xt', xt', freepar(i));
    end
    if params.vfold
        for f = 1:params.vfold,
            in  = find(params.groups ~= f);
            out = find(params.groups == f);
            if params.svm_type == 2
                % OC-SVM has only target samples for training
                in = in((yt(in) == params.target));
            end
            if params.knl_type == 4
                model = svmtrain(yt(in),kt(in,in), libpar);
                yp = svmpredict(yt(out),kt(out,in), model);
            else
                model  = svmtrain(yt(in),xt(in,:), libpar);
                yp = svmpredict(yt(out),xt(out,:), model);
            end
            %if verb, plot(1:length(yp),yp,1:length(out),yt(out,:)), axis('tight'), shg, end
            s = warning('off','MATLAB:divideByZero');
            assess = assessment(yt(out,:),yp,assespar);
            warning(s);
            if params.svm_type < 3
                error(f,i) = assess.Kappa;
            else
                error(f,i) = assess.RMSE; %sqrt(mean((yt(out,:)-yp).^2));
            end
        end
        %if verb, fprintf('    mean K/R: %f\n', mean(error(:,i))), end
    else
        if params.svm_type == 2
            % OC-SVM has only target samples for training
            ct = ct(yt(params.ct) == params.target);
        else
            ct = params.ct;
        end
        if params.knl_type == 4
            model = svmtrain(yt(ct,:), kt(ct,ct), libpar);
            yp = svmpredict(yt(params.cv,:), kt(params.cv,ct), model);
        else
            model = svmtrain(yt(ct,:), xt(ct,:), libpar);
            yp = svmpredict(yt(params.cv,:), xt(params.cv,:), model);
        end
        %if verb, plot(1:length(yp),yp,1:length(cv),yt(cv,:)), axis('tight'), shg, end
        s = warning('off','MATLAB:divideByZero');
        assess = assessment(yt(params.cv,:),yp,assespar);
        warning(s);
        if params.svm_type < 3
            error(i) = assess.Kappa;
        else
            error(i) = assess.RMSE; %sqrt(mean((yt(cv,:)-yp).^2));
        end
        %if verb, fprintf('    SV: %d, K/R: %f\n', model.totalSV, error(i)), end
    end
end

if params.svm_type < 3 % SVC
    if params.vfold
        [val,ifreepar] = max(mean(error,1));
    else
        [val,ifreepar] = max(error);
    end
else                   % SVR
    if params.vfold
        [val,ifreepar] = min(mean(error,1));
    else
        [val,ifreepar] = min(error);
    end
    %semilogx(ifreepar,mean(error,1)), xlabel('Param'), ylabel('Kappa/RMSE'), axis('tight'), drawnow
end

if verb, fprintf('    selected: %f, K/R %f\n', freepar(ifreepar), val), end
