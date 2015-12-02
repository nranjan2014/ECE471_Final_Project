#plot1
    x1=c(1,0.479167,0.3125,0.229167,0.208333,0.166667,0.145833,0.125,0.104167,0.104167,0.104167,0.104167,0.104167,0.104167,0.0625,0.0625,0.0416667,0.0416667,0.0416667,0.0208333,0)
    y1=c(1,0.945578,0.918367,0.870748,0.843537,0.823129,0.809524,0.789116,0.77551,0.761905,0.748299,0.748299,0.727891,0.693878,0.673469,0.646259,0.619048,0.591837,0.571429,0.537415,0)


    x3=c(1,0.77551,0.734694,0.673469,0.632653,0.612245,0.571429,0.530612,0.510204,0.510204,0.510204,0.510204,0.489796,0.489796,0.469388,0.469388,0.428571,0.326531,0.285714,0.244898,0)
    y3=c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.993151,0.993151,0.979452,0.965753,0.952055,0.89726,0)


    png(file="plotcombined20.png")

    plot(x1, y1, main="ROC curve", xlab="FPR", ylab="TPR", type='b', pch=15)
    points(x1,y1,col="red", type='b', pch=15)
    points(x3,y3,col="green", type='b', pch=15)


    legend("right", inset=.05, title="",
            c("tX","fX"), fill=c("red", "green"), horiz=FALSE)

dev.off()

#plot1 ends




    png(file="plottx20.png")
    plot(x1, y1, main="ROC curve", xlab="FPR", ylab="TPR", type='b', pch=15, col="red")
    legend("right", inset=.05, title="",
            c("tX"), fill=c("red"), horiz=FALSE)
dev.off()



    png(file="plotfx20.png")
    plot(x3, y3, main="ROC curve", xlab="FPR", ylab="TPR", type='b', pch=15, col="green")
    legend("right", inset=.05, title="",
            c("fX"), fill=c("green"), horiz=FALSE)
dev.off()







