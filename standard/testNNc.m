function Ypred_NN = testNNc(net,XXtotal)

Ypred_NN = sim(net,XXtotal');
[~,Ypred_NN] = max(Ypred_NN);
Ypred_NN = Ypred_NN';
