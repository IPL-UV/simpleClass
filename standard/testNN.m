function Ypred_NN = testNN(net,XXtotal)

Ypred_NN = sim(net,XXtotal');
[val Ypred_NN] = max(Ypred_NN);
Ypred_NN = Ypred_NN';