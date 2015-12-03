% ECE471/571
% How to illustrate data points
%

% clear the figure
clf;

% load the training set
load data2NewKMNorm.dat;
Tr = data2NewKMNorm;

% extract the samples belonging to different classes
  % find the row indices for the 1st class, labeled as 0
Tr0 = Tr(:,[1:2]);
save Tr0;                % so that you can use it directly next time 


% plot the samples
plot(Tr0,'r*'); % use "red" for class 0
hold on;           % so that the future plots can be superimposed on the previous ones