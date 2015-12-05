% Source code: Matlab Code for Implementing  BPNN: 

%A, B, C, D, E, and F are our features derived from PCA
%STATUS is the ground truth classifications 

X = [A B C D E F]
T = transpose(STATUS);
P = X’
net = feedforwardnet([10 9 8 7], 'trainlm')
net = configure(net, P, T)
net.trainParam.goal = 1e-8
net.trainParam.epochs = 1000
net = train(net, P, T)
view(net)
y = net(P)
perf = perform(net, T, y)
Y = sim(net, P)

