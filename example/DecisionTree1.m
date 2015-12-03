% ECE471/571 project 1
% How to illustrate data points
%

% clear the figure
clf;

% load the training set
%load NormData.tr;
%Tr = NormData;
%Tr2 = Tr(:,23);
%save Tr2
%Tr1 = Tr(:, [1:22]);
%save Tr1; 

%load PCA.tr;
%Tr = PCA;
%Tr2 = Tr(:,7);
%save Tr2
%Tr1 = Tr(:, [1:6]);
%save Tr1;

load FLD.tr;
Tr = FLD;
Tr2 = Tr(:,2);
save Tr2
Tr1 = Tr(:,1);
save Tr1;

%ctree = fitctree(Tr1,Tr2,'CrossVal', 'on'); % create classification tree
tree = fitctree(Tr1,Tr2);
cvmodel = crossval(tree);
%view(ctree.Trained{1}, 'Mode', 'graph'); % text description
view(tree, 'Mode', 'graph'); % text description
L = kfoldLoss(cvmodel);
L1 = kfoldLoss(cvmodel, 'mode','individual');

    
    
   