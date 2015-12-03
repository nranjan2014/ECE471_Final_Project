library(cluster)
library(fpc)

data(PCA.tr)
dat <- PCA[, -5] # without known classification 
# Kmeans clustre analysis
clus <- kmeans(dat, centers=2)
