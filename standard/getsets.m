function [ct,cv] = getsets(YY, classes, percent, vfold, par, [randstate])

% function [ct,cv] = getsets(YY, classes, percent, vfold, par, [randstate])
% This funcion return a train set (ct) and a validation set (cv) among the
% classes 'classes' of vector 'YY'. 'percent' is the length percentage of
% the 'cv' set, and the optional argument 'randstate' sets the state of the
% random generator (by default = 0). If 'vfold' is nonzero, then a cross
% validation method is to be used and we just make sets with all the samples.
%
% 'par' must be either 'class' or 'regress'. In 'class' mode either vfold or
% percent can be used as explained. In regress mode only regress is used. If
% not specified 'regress' mode is used by default.
%
% By JoRdI 2006-07
%    2016: Added 'par' for 'class' or 'regress' modes.

if ~exist('par', 'var')
    par = 'regress'
    disp('INFO: using regression as learning paradigm for division')
end

if ~exist('randstate', 'var')
    randstate = 0;
end

s = rand('state');
rand('state',randstate);

ct = [];
cv = [];

switch lower(par)
    case 'class'
        for ii = 1:length(classes)
            idt = find(YY == classes(ii));
            if vfold
                ct = union(ct,idt);
                cv = ct;
            else
                lt  = fix(length(idt) * percent);
                idt = idt(randperm(length(idt)));
                %idv = idt(1:end-lt);  % remove cue
                idv = idt(lt+1:end); % remove head
                idt = setdiff(idt,idv);
                ct  = union(ct,idt);
                cv  = union(cv,idv);
            end
        end
        
case 'regress'
    if percent < 0 || percent >= 1
        error(['Invalid value for percent, must be in [0,1]: ' num2str(percent)])
    end
    idt = 1:length(YY);
    lt  = fix(length(idt) * percent);
    idt = idt(randperm(length(idt)));
    idv = idt(lt+1:end);
    idt = setdiff(idt,idv);
    ct  = union(ct,idt);
    cv  = union(cv,idv);
    
otherwise
    disp('Unknown learning paradigm, must be ')
end

rand('state',s);

