function groups = folds(n, vf)

% function groups = folds(n, vf)
%
% Creates the vf groups for a length n

s = rng;
rng(0);
groups = ceil(vf * randperm(n) / n);
rng(s);
