
P = [0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1; 0 0; 0 1; 1 0; 1 1]';
T = [0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0]; % desired output
net = feedforwardnet([2 3],'trainlm' );  
net = configure(net, P, T);
net.trainParam.goal = 1e-8;
net.trainParam.epochs = 1000;
net = train(net, P, T);
view(net)
y = net(P);
perf = perform(net,T,y) % performance
Y = sim(net, P)  % result from simulation
