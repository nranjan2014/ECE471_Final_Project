% ECE471/571 project 1
% How to illustrate data points
%

% clear the figure
clf;

% load the training set
load PCA.tr;

Tr = PCA;

Tr2 = Tr(:,7);

save Tr2

Tr1 = Tr(:, [1:6]);
save Tr1; 


%index = crossvalind('kfold',Tr2, 10);
%cp = classperf(Tr2);
%for i = 1:10
   % test = (index==i);
    %train = ~test;
    
    ctree = fitctree(Tr1,Tr2,'CrossVal', 'on'); % create classification tree
    view(ctree.Trained{1}, 'Mode', 'graph'); % text description
    %resuberror = resubLoss(ctree);
    %sfit1 = eval(ctree);
   % y = predict(ctree,test);
   % index = cellfun(@strcmp, y, Tr2(test));
   % errorMat(i) = sum(index)/length(y);
    %classperf(cp,ctree,test)
    
%end
%cp.errorRate