 
%%%%%% Scale before calling TrainSVC_crit%%%%%
function  [f]=scale(x)

[samples features] = size(x);

for j=1:features%j counts features:1,2,..d
   mu=min(x(:,j));
   sig=max(x(:,j))-min(x(:,j));
   f(:,j)=(x(:,j)-mu)/sig;% scaling each feature to the unit range 
end
 
 