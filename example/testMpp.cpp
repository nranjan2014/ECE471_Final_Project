
/***********************************************************************************************************************
*
* ECE 471 : Final Project
* Code developed from implementation from Project 1, 2, 3, and 4
* Last Modified by/Date: Niloo Ranjan, 11/19/2015
* Addition: modified the command line argument option. Auugmented the original file with functions and code
* needed to perform the tasks for final project.
* Added cases options to perform other tasks of the final project.
*
* Case 1: get the Normalized, PCA, FLD, 10-fold done
* Case 2: MAP case1 classification on nX data set with equal prior probabilty
* Case 3: MAP case2 classification on nX data set with equal prior probabilty
* Case 4: MAP case3 classification on nX data set with equal prior probabilty
* Case 5: finding optimal prior probability using MPP case 3 with PCA data
* Case 6: classification using MAP on PCA data set
* Case 7: classification using MAP on FLD data set
* Case 8: getting FPR TPR for ROC Curve plot for PCA data set
* Case 9: getting FPR TPR for ROC Curve plot for FLD data set
* Case 10: getting FPR TPR for ROC Curve plot for Normalized data set
* Case 11: kNN implementation using full Euclidean distance on Normalized data set
* Case 12: kNN implementation using full Euclidean distance on PCA data set
* Case 13: kNN implementation using partial Euclidean distance on FLD data set
* Case 14: K-means implementation on Normalized data
* Case 15: Winner Take All implementation on Normalized data
* Case 16: Classifier fusion Naive Bayes
* Case 17: make label from K-means classification similar to class label for sample from classifier using 10-fold data
*************************************************************************************************************************/

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include "Matrix.h"             
#include "Pr.h"
#include <ctime>

using namespace std;

#define Usage "Usage: ./testMpp training_set classes features cases K-vaues \n\t training_set: the file name for data set\n\t classes: number of classes\n\t features: number of features (dimension)\n\t cases: used to run diffetent task categories, cases can be from 1 to 17\n\t K-values: for kNN \n\n"

int main(int argc, char **argv)
{
    int nrTr;       // number of rows in the training and test set
    int  nc;              // number of columns in the data set; both the 
    // training and test set should have the same
    // column number
    Matrix XTr;      // original training and testing set
    Matrix Tr;        // training and testing set file received from command line
    Matrix tXTr;
    Matrix fXTr;
    Matrix nXTr;
    
    // check to see if the number of argument is correct
    if (argc < 6) {
        cout << Usage;
        exit(1);
    }

    int classes = atoi(argv[2]);   // number of classes
    int nf = atoi(argv[3]);        // number of features (dimension)
    int cases = atoi(argv[4]);     // number of features (dimension)

    int K = atoi(argv[5]);
    
    // read in data from the data file
    nc = nf+1; // the data dimension; plus the one label column

    XTr = readData(argv[1], nc);
    nrTr = XTr.getRow();          // get the number of rows in training set
  //  cout << "row: " << nrTr << endl;
    Matrix X;
if ( cases == 1)
{
    X = subMatrix(XTr, 0,0, nrTr-1, nf-1);

    Matrix means, covr, covarxN, sigma, nXTr,D_M, V_M, B_M, E_M  ;
    // B_M basis vector
    // E_M eigen vector corresponding to eigen values error rate less than 0.10

    // calculate the mean and the variance of the original data set as whole without class
    // consideration
    sigma.createMatrix(nf, 1);
    means = mean(X,nf);
    covr = cov(X, nf);

    nXTr = XTr;
    // normalize the traning data set with sigma from covariance and mean already calculated
    for (int j = 0; j < nf ; j++)
    {
        sigma(j,0) = sqrt(covr(j,j));
        for (int i =0; i < nrTr; i++)
        {
            nXTr(i,j) = (nXTr(i,j) -  means(j,0)) / sigma(j,0);

        }
    }
    writeData(nXTr, "NormData.tr");
    cout << "Column NormData: " << nXTr.getCol() << endl;
    // perform the transformation of normalized data set using PCA
    Matrix tXTr, tXTe;
   
    // get optimal basis vector
    B_M = GetPCA(nXTr);
    
    // using this basis vector to transformed training and testing data set
    tXTr = GetDataTransformedFLD ( nXTr, B_M);
    
    writeData(tXTr, "PCA.tr");
    cout << "Column PCA.tr: " << tXTr.getCol() << endl;

    // perform the transformation of normalized data set using FLD
    Matrix W, fXTr, fXTe;
    // get the optimal projection direction W
    W = GetFLD(nXTr);

    // using this optimal projection direction to transformed training and testing data set
    fXTr = GetDataTransformedFLD ( nXTr,W);
    
    writeData(fXTr, "FLD.tr");
    cout << "Column FLD.tr: " << fXTr.getCol() << endl;
    
    MakeFoldData(nrTr);
    
    int fea = 22;   // there are 9 features in this data set
    int sumError = 0;
    int sumTotalCount = 0;
    double totalTime = 0.0;
    
    int sampFold = 20;
}
    //run the task of  MAP case1 nX data set with equal prior probabilty
    else if ( cases == 2)
    {
        nXTr = readData("NormData.tr", 23);
        RunMPPCase123 (nXTr, 1);
    }
    else if ( cases == 3)
    {
        // run the task of  MAP case2 classification on nX data set with equal prior probabilty
        nXTr = readData("NormData.tr", 23);
        RunMPPCase123 (nXTr, 2);
    }
    else if ( cases == 4)
    {
    //run the task of  MAP case3 classification on tX (PCA) data set with equal prior probabilty
       tXTr = readData("PCA.tr", 7);
       RunMPPCase123 (tXTr, 3);
    }
    // perform the task of finding optimal prior probability

    else if ( cases == 5)
    {
        tXTr = readData("PCA.tr", 7);
        FindOptimalPP( tXTr);
    }
    // perform the task of classification using MAP on data set tX
    else if ( cases == 6)
    {
        tXTr = readData("PCA.tr", 7);
        ClassifyBestPwPCAFLD(tXTr);
    }
    // perform the task of classification using MAP on data set fX
    else if ( cases == 7)
    {
       fXTr = readData("FLD.tr", 2);
       ClassifyBestPwPCAFLD(fXTr);
    }
    // perform the task of getting FPR TPR for ROC Curve plot for tX data set
    else if ( cases == 8)
    {
        tXTr = readData("PCA.tr", 7);
        GetROC(tXTr);
    }
    // perform the task of getting FPR TPR for ROC Curve plot for fX data set
    else if ( cases == 9)
    {
        fXTr = readData("FLD.tr", 2);
        GetROC(fXTr);
    }
    // perform the task of getting FPR TPR for ROC Curve plot for nX data set
    else if ( cases == 10)
    {
        nXTr = readData("NormData.tr", 23);
      //  GetROC(nXTr);
       // tXTr = readData("PCA.tr", 7);
        ClassifyBestPwPCAFLD(nXTr);

    }
    // kNN implementation using full Euclidean distance on data set nX
    else if ( cases ==11)
    {
        nXTr = readData("NormData.tr", 23);
        
        RunKNN (nXTr, K);
    }

    // kNN basic implementation using full Euclidean distance on data set tX
    else if ( cases == 12)
    {
        tXTr = readData("PCA.tr", 7);
        RunKNN (tXTr, K);

    }
    // kNN implementation using full Euclidean distance on data set fX
    else if ( cases == 13)
    {
       fXTr = readData("FLD.tr", 2);
       RunKNN (fXTr, K);
    }
    // K-means implementation
    else if ( cases == 14)
    {
        
        nXTr = readData("NormData.tr", 23);
        
        RunKmeansClustering(nXTr);

    }
    // winner take all implementation
    else if ( cases == 15)
    {
        nXTr = readData("NormData.tr", 23);
        
        RunWTAClustering(nXTr);

    }
    // classifier fusion Naive Bayes
    else if ( cases == 16)
    {
        RunNaiveBayes();
    }
    // make label from K-means classification similar to class label
    // for sample from classifier using 10-fold data
    else if ( cases == 17)
    {
        RunClusterLabel();

    }
    else
    {
        cout << "Please specify the case from 1 to 17, which task to run" << endl;
    }

    return 0;
}


// this function calculates the performance metrics for the classification rule MAP used
// on the each type of data set tested
void DerivePerformanceMetric ( Matrix & tested, int datatype)
{
    double Sensitivity;
    double Specificity;
    double Precision;
    double Recall;
    double Accuracy;
    double TPR;
    double FPR;
    int CorrectCountMPP = 0;
    int TP; // true positive number
    int TN; // true negative number
    int FP; // false positive number
    int FN; // false negative number
    TN = 0;
    TP = 0;
    FP = 0;
    FN = 0;
    Sensitivity = 0.0;
    Specificity = 0.0;
    Precision = 0.0;
    Recall = 0.0;
    Accuracy = 0.0;
    TPR =0.0;
    FPR = 0.0;
    int row = tested.getRow();
    int col = tested.getCol();
    Matrix ConfusionMatrix (2,2);
    
    for (int i=0; i<row; i++) {
        if (tested(i,0) == tested(i,1))
        {
            CorrectCountMPP++;
            if ( tested(i,0) == 1) // truth yes, observed yes
            {
                TP++;
            }
            else    // truth no, observed no
            {
                TN++;
            }
        }
        else
        {
            if (tested(i,0) == 1) // truth yes, observed no
            {
                FN++;
            }
            else    // truth no, observed yes
            {
                FP++;
            }
        }
    }
    
    Sensitivity = ((double)(TP))/(TP+FN);
    Specificity = ((double)(TN))/(TN+FP);
    Precision = ((double)(TP))/(TP+FP);
    Recall = ((double)(TP))/(TP+FN);
    Accuracy = ((double)((TP+TN)))/(TP+TN+FN+FP);
    FPR = ((double)(FP))/(FP+TN);
    TPR = ((double)(TP))/(TP+FN);
    ConfusionMatrix(0,0) = TP;
    ConfusionMatrix(0,1) = FN;
    ConfusionMatrix(1,0) = FP;
    ConfusionMatrix(1,1) = TN;
    
    string lb = "ConfusionMatrix" + to_string(datatype);
  //  string lb = "ConfusionMatrixMPPCase3PCA" + to_string(datatype);
   // string lb = "ConfusionMatrixMPPCase3FLD" + //to_string(datatype);
   // string lb = "ConfusionMatrixKNNNorm" + to_string(datatype);
    //string lb = "ConfusionMatrixKNNPCA" + to_string(datatype);
   // string lb = "ConfusionMatrixKNNFLD" + to_string(datatype);
  //  string lb = "ConfusionMatrixNormOptPP" + to_string(datatype);
    //string lb = "ConfusionMatrixKmeans" + to_string(datatype);
    
    std::string str = "string";
    const char *cstr = lb.c_str();

    writeData(ConfusionMatrix, cstr);
    
    cout << "The TP rate using " << datatype << " is " << TP << endl;
    cout << "The TN rate using " << datatype << " is " << TN << endl;
    cout << "The FP rate using " << datatype << " is " << FP << endl;
    cout << "The FN rate using " << datatype << " is " << FN << endl;
    cout << "The Sensitivity rate using " << datatype << " is " << Sensitivity << endl;
    cout << "The Specificity rate using " << datatype << " is " << Specificity << endl;
    cout << "The Precision rate using " << datatype << " is " << Precision << endl;
    cout << "The Recall rate using " << datatype << " is " << Recall << endl;
    cout << "The Accuracy rate using " << datatype << " is " << Accuracy << endl;
    
    cout << "The FPR rate using " << datatype << " is " << FPR << endl;
    
    cout << "The TPR rate using " << datatype << " is " << TPR << endl;
    
}

// this function finds the optimal projection direction for FLD
Matrix GetFLD ( Matrix &train)
{
    static int nf;

    static Matrix *means, *covs;

    int nctr, nrtr;

    int i, j, n, c;

    Matrix SW, invSW, ret, tmp;

    nctr = train.getCol();

    nrtr = train.getRow();

    nf = nctr-1;

    c = 2;

    means = (Matrix *) new Matrix [c];

    for (i=0; i<c; i++)
        means[i].createMatrix(nf, 1);

    covs = (Matrix *) new Matrix [c];

    for (i=0; i<c; i++)
        covs[i].createMatrix(nf, nf);

    // get means and covariance of the classes
    for (i=0; i<c; i++) {
        tmp = getType(train, i);
        n = tmp.getRow();
        // get sw1 and sw2 sw = (n-1) covariance
        covs[i] = (n-1) * cov(tmp, nf);
        // find m1 and m2
        means[i] = mean(tmp, nf);
    }
    SW.createMatrix(nf, nf);

    // get sw = sw1 + sw2
    for (i=0; i < c; i++) {
        SW = SW + covs[i];
    }

    invSW = inverse(SW);

    // find W = sw+inverse + ( m1 - m2)
    ret = invSW->*(means[0] - means[1]);

    return ret;
}
// this function transforms the normalized training and the testing set
// W = optimal projection direction for FLD case
// W = basis vector for PCA case
Matrix GetDataTransformedFLD ( Matrix &nX, Matrix W)
{
    Matrix temp3, temp4, fx, fxte;
    int row;
    int col;

    row = nX.getRow();
    col = nX.getCol();

    temp3 = subMatrix(nX, 0, 0,row-1, col-2 );

    // for FLD it is y = W_transpose * X
    // for PCA it is y = B_transpose * X
    temp3 = transpose(W) ->* transpose( temp3);
    temp3 = transpose(temp3);

    fx.createMatrix(row, temp3.getCol()+1);

    // add the class label to the transformed data set
    for ( int i =0; i < row; i++ )
    {
        int j;
        for ( j =0; j < temp3.getCol(); j++)
        {
            fx(i,j) = temp3(i,j);
        }
        fx (i, j) = nX(i, col-1);
    }

    return fx;
}

// this function calculets the eigen values, eigen vector, basis vector for PCA 
Matrix GetPCA( Matrix &nX)
{

    Matrix temp, D_M, V_M, B_M, E_M, covarxN ;
    temp = nX;
    int row;
    int col;

    row = nX.getRow();
    col = nX.getCol();


    D_M.createMatrix(1,col-1);
    V_M.createMatrix(col-1,col-1);
    covarxN.createMatrix(col-1, col-1);

    temp = subMatrix(nX, 0,0, row-1, col-2);
    covarxN = cov(temp, col-1);

    // find the eigen values and eigen vector from the covariance matrix
    // of the normalized data set
    jacobi(covarxN, D_M, V_M );

    // sort the eigen values and rearrange the eigen vector accordingly
    eigsrt(D_M, V_M);

    double Sum_D_M = 0;
    double D_M_add = 0;
    int count =0;
    int eigncol = V_M.getCol();
    int eignrow = V_M.getRow();
    int i, j;

    // get the sum of all eigen values
    for ( i =0; i < eigncol; i++)
    {
        Sum_D_M += D_M(0,i);
    }

    // keep the eigen values that will give eoor rate less than 0.10
    for (  j = 0; j < eigncol; j++)
    {
        D_M_add += D_M(0,j);

        if((D_M_add / Sum_D_M) > 0.10)
        {
            break;
        }
    }
    
    // basic vector with higher values of eigen values
    B_M = subMatrix(V_M, 0, j, eignrow-1, eigncol-1);
    E_M = subMatrix(D_M, 0, j, 0, eigncol-1);
   // cout << "eigen_vector" << E_M << endl;

    return B_M;
}

// this function implement the basic implementattion of the kNN using 
// original Euclidean distance to be used to clasiify the testing set of 
// nX, tX, fX data set which is normalized, PCA transformed, FLD trasformed respectively	
int KNNClassifierEuclidian(Matrix &nXTr, Matrix &sample, int K)
{
    Matrix distance(nXTr.getRow(),2);
    Matrix samp;
    double sumDistance = 0.0;

    samp = subMatrix(sample, 0,0,0,sample.getCol()-1);

    // calculate the original Euclidean distance between the current tesing sample and each of the 
    // sample in the training set
    for ( int i =0; i < nXTr.getRow(); i++)
    {
        sumDistance = 0.0;
        Matrix samp1 = subMatrix(nXTr, i, 0,i,nXTr.getCol()-2);

        // get the square of the difference distance for each feature of the 
        for ( int j =0; j < samp1.getCol(); j++)
        {
            sumDistance = sumDistance + pow((samp(0,j) - samp1(0,j)), 2);
        }

        sumDistance = sqrt(sumDistance);

        // keep the disance calculated from each training sample along 
        // with their class label info
        distance(i,0) = sumDistance;
        distance(i,1) = nXTr(i, nXTr.getCol()-1);
    }

    // get just the distance part as row to use insersort function to sort the distances
    Matrix DisUnSort = transpose(subMatrix(distance, 0, 0, distance.getRow()-1, 0));

    Matrix DisSort(DisUnSort.getRow(), DisUnSort.getCol()); // this will keep sorted distance
    Matrix DisPos(DisUnSort.getRow(), DisUnSort.getCol());  // this will keep the index 
    // of the sorted distance 

    insertsort(DisUnSort, DisSort, DisPos);
    Matrix Kdistance(K, 2); // to keep only K nearest distances

    // get only the k nearest distance from the list of all calcualted sorted 
    // distance along with the label for the sample corresponding at that distance apart
    for ( int i =0; i< K; i++)
    {
        int pos = DisPos(0,i);
        int label = distance(pos,1);
        Kdistance(i,0) = DisSort (0,i);
        Kdistance(i,1) = label;
    }

    int maxNum = 0;
    int Class = 0;

    // as in this case there are only two class category 0 or 1 in the 
    // get the number of sample fall withing that K nearest neighborhood 
    // for each type and assign the class corresponding to the class with 
    // max number of sample found 
    Matrix Type0 = getType(Kdistance,0);
    Matrix Type1 = getType(Kdistance,1);
    if (Type0.getRow() > Type1.getRow())
    {
        Class= 0;
    }
    else
    {
        Class = 1;
    }
    return(Class);
}

// this method divides the whole data set into 10 parts
// to be used for 10 fold cross validation for testing
void MakeFoldData(int nrTr)
{
    std::srand(std::time(NULL));
    unsigned lb = 0, ub = 194;
    Matrix fold(10, 20);
    fold.initMatrix(-1);
    Matrix fl(200,1);
    fl.initMatrix(-1);
    bool fill = true;
    int cot =0;
    int h =0;
    do
    {
        fill = true; // to fill with unique sample in each fold
        
        // get the random number between 0 and 194
        unsigned rnd = lb + (rand() % ((ub - lb) + 1));
        
        // make the 10 fold each with 20 samples in it
        for ( int k = 0; k < 200;k++)
        {
            if ( fl(k,0) == rnd)
            {
                fill = false;
                break;
            }
        }
        if (fill)
        {
            fl(h,0) = rnd;
            h++;
        }
        cot++;
    }while(h < nrTr);
    int k =0;

    for ( int i =0; i < 10; i++)
    {
        for ( int j =0; j < 20; j++)
        {
            fold(i,j) = fl(k,0);
            k++;
        }
        
    }
    writeData(fold, "foldTen.dat");
    cout << fold << endl;
}

// this method build the validating data set for 10-fold cross-validation
Matrix getTestingData(Matrix &S, Matrix &glassData)
{
    int row = 0;
    int row1 = -1;
    
    // ge the count of row number without -1 value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire "fglass" data set
    for ( int i =0; i < S.getCol(); i++)
    {
        int n = S(0,i);
        
        // get how many row would be in the validating set
        // disregard the -1 as it was to make matrix holding
        // fold to be of same dimention
        if ( n != -1)
        {
            row++;
        }
    }
    
    Matrix ret(row, glassData.getCol());
    
    // ge the row number without -1 value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire data set
    for ( int i = 0; i < S.getCol(); i++)
    {
        int n = S(0,i);
        
        if ( n != -1)
        {
            row1++;
            
            Matrix Samp = subMatrix(glassData,n,0,n,glassData.getCol()-1);
            
            for ( int j = 0; j <Samp.getCol(); j++  )
            {
                // include the corresponding sample from the data set
                ret(row1,j) = Samp(0,j);
            }
        }
        
    }
    
    return ret;
    
}

// this method build the training data set for 10-fold cross-validation
Matrix getTrainingData(Matrix &foldData, Matrix &glassData, int i )
{
    Matrix samp;
    int R1 = 0;
    
    // i indicates the fold number that is being consider for serving as validating
    // set in current round of toal of 10 rounds.
    
    // in this case i is the first fold so include all other 9 fold excluding
    // first fold to build the training set at current round of toal of 10 rounds.
    if ( i == 0)
    {
        samp = subMatrix(foldData,i+1,0,foldData.getRow()-1,foldData.getCol()-1);
    }
    
    // in this case i is the last fold so include all other 9 fold excluding
    // last fold to build the training set at current round of toal of 10 rounds.
    else if ( i == foldData.getRow()-1)
    {
        samp = subMatrix(foldData,0,0,foldData.getRow()-2,foldData.getCol()-1);
    }
    // in this case i is anywhre between first and last fold number.
    else
    {
        // so first half of the fold from all fold above current fold
        Matrix first = subMatrix(foldData,0,0,i-1,foldData.getCol()-1);
        
        // and second half  from all fold below current fold,
        Matrix second = subMatrix(foldData,i+1,0,foldData.getRow()-1,foldData.getCol()-1);
        
        //then combine these partial fold to get one whole fold
        Matrix whole(first.getRow()+second.getRow(), first.getCol());
        
        int R = 0;
        //use this whole fold to build the training set at current round of toal of 10 rounds.
        for ( R = 0; R < first.getRow(); R++)
        {
            for ( int j =0; j< first.getCol(); j++)
            {
                whole(R,j) = first(R,j);
            }
        }
        while ( R1 < second.getRow() )
        {
            for ( int j =0; j< second.getCol(); j++)
            {
                whole(R,j) = second(R1,j);
            }
            R++;
            R1++;
        }
        samp = whole;
    }
    
    int row = 0;
    int row1 = -1;
    
    // ge the count of row number without -1 value of the sample from the current fold pass that needs to be
    // included in this validating set among the entire "fglass" data set
    for ( int i = 0; i < samp.getRow(); i++)
    {
        for (int j = 0; j < samp.getCol(); j++)
        {
            int n = samp(i,j);
            if ( n != -1)
            {
                row++;
            }
        }
    }
    
    Matrix ret(row, glassData.getCol());
    
    // ge the row number without -1 value of the sample from the current fold pass that needs to be
    // included in this validating set among the data set
    for ( int i = 0; i < samp.getRow(); i++)
    {
        for (int j =0; j < samp.getCol(); j++)
        {
            int n = samp(i,j);
            if ( n != -1)
            {
                row1++;
                
                Matrix Samp1 = subMatrix(glassData,n,0,n,glassData.getCol()-1);
                
                for ( int k = 0; k < Samp1.getCol(); k++  )
                {
                    ret(row1,k) = Samp1(0,k);
                }
            }
        }
    }
    return ret;
}

// this function implements the MPP for case 1, 2, and 3
// with equal prior probability
void RunMPPCase123 (Matrix &nXTr, int caseNum)
{
    int fea = 22;   // there are 22 features in this data set
    int sumCorrect = 0;
    int sumTotalCount = 0;
    double totalTime = 0.0;
    
    int sampFold = 20;
    int numrow = nXTr.getRow();
    int classes = 2;
    Matrix labelTotal(numrow, 2);
    
    // read in fold data
    Matrix foldData = readData("foldTen.dat", sampFold );
    
    // read one row at a time from the 10 folds and assign the current row read as
    // fold for building validating set. Then assign rest of the current to use as
    // training set using kNN classifier c;assify the validating set from training set
    int labelrow = 0;
    for (int j = 0; j < foldData.getRow(); j++ )
    {
        // current fold
        Matrix S = subMatrix(foldData, j,0,j,foldData.getCol()-1);

        // build the validating set from the one fold just read
        Matrix test = getTestingData(S, nXTr);

        // build the training set from leaving the current fold and including the rest of the
        // fold
        Matrix training = getTrainingData(foldData, nXTr, j );

        // to hold class label for testing samples
        Matrix label (test.getRow(), 2);
        
        
        int foldclass = 2;
        int CorrectCount = 0;
        
        // assign prior probability
        Matrix Pw(classes, 1);
        for (int i=0; i<classes; i++)
            Pw(i,0) = (float)1/classes;   // assuming equal prior probability
        
        
        // start timming the kNN classification completion for one fold
        // of the validating set
        clock_t start = clock();
        
        
        
        // for each sample in the validating set use the kNN
        // with original Euclidean disable to find the class label of the sample
        for (int i = 0; i < test.getRow(); i++)
        {
            Matrix sample = transpose(subMatrix(test, i,0,i, test.getCol()-2));
            
            // call MPP to perform classification
            label(i,0) = test(i,test.getCol()-1);
            label(i,1) = mpp(training, sample, classes, caseNum , Pw);
            
            
            labelTotal(labelrow, 0) = test(i,test.getCol()-1);
            labelTotal(labelrow, 1) = label(i,1);
            labelrow++;
            
            if (label(i,1) == test(i,test.getCol()-1))
            {
                CorrectCount++;
            }
        }

        
        sumCorrect = sumCorrect + CorrectCount;
        sumTotalCount = sumTotalCount + test.getRow();
        
        cout << "Correct Rate: " << (((float) CorrectCount) / test.getRow())*100 << endl;
        
        clock_t end = clock();
        totalTime = totalTime + (((double) (end-start)) / 1000000);
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
   if (caseNum == 1)
   {
    writeData(labelTotal, "labelTotalMPP1.dat");
    // get the performance metrics for the classification tested
    DerivePerformanceMetric ( labelTotal, 1);
   }
    else if (caseNum == 2)
    {
        writeData(labelTotal, "labelTotalMPP2.dat");
        DerivePerformanceMetric ( labelTotal, 2);
    }
    else if (caseNum == 3)
    {
        writeData(labelTotal, "labelTotalMPP3.dat");
        DerivePerformanceMetric ( labelTotal, 3);
    }
    
    
    cout << "Average Correct Rate: " << (((float) sumCorrect) / sumTotalCount)*100 << endl;
    cout << "Average Running Time: " << totalTime / 10 << " seconds" << endl;
    
    
}

// this function finds the optimal prior probability
// to use for MPP case 1,2,3
void FindOptimalPP( Matrix &nXTr)
{
    double maxMPP = 0.0;
    double accuracy = 0.0;
    Matrix highPw(2,1);
    Matrix Pw(2, 1);

    int fea = 22;   // there are 22 features in this data set
    double sumCorrect = 0;
    double sumTotalCount = 0;
    double totalTime = 0.0;
    double avg = 0.0;
    int sampFold = 20;
    int classes = 2;
    
    // read in fold data
    Matrix foldData = readData("foldTen.dat", sampFold );
    

    // varied the range of prior probabilty over 0 to 1 with increament of 0.1
    for (int j = 0 ; j < 11; j++)
    {
        Pw(0,0) = (float)j/10;
        
        Pw(1,0) = 1 - (Pw(0,0));
        
        

        for (int i = 0; i < foldData.getRow(); i++ )
        {
            // current fold
            Matrix S = subMatrix(foldData, i,0,i,foldData.getCol()-1);

            // build the validating set from the one fold just read
            Matrix test = getTestingData(S, nXTr);

            // build the training set from leaving the current fold and including the rest of the
            // fold
            Matrix training = getTrainingData(foldData, nXTr, i );

            // to hold class label for testing samples
            Matrix label (test.getRow(), 1);
            
            
            int foldclass = 2;
            double CorrectCount = 0;
            
            // start timming the classification completion for one fold
            // of the validating set
            clock_t start = clock();
            
            
            // for each sample in the validating set use MPP
            // to find the class label of the sample
            for (int i = 0; i < test.getRow(); i++)
            {
                Matrix sample = transpose(subMatrix(test, i,0,i, test.getCol()-2));
                
                // call MPP to perform classification
                label(i,0) = mpp(training, sample, classes, 2, Pw);
                
                if (label(i,0) == test(i,test.getCol()-1))
                {
                    CorrectCount++;
                }
            }
            
            sumCorrect = sumCorrect + CorrectCount;
            sumTotalCount = sumTotalCount + test.getRow();
            
            cout << "Correct Rate: " << (((float) CorrectCount) / test.getRow())*100 << endl;
            
            clock_t end = clock();
            totalTime = totalTime + (((double) (end-start)) / 1000000);
            cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
        }

        avg = (((float) sumCorrect) / sumTotalCount)*100;
        cout << "Average correct Rate: " << (((float) sumCorrect) / sumTotalCount)*100 << endl;
        cout << "Average Running Time: " << totalTime / 10 << " seconds" << endl;
        
        // keep track of max accuracy and corresponding prior probability found so far
        if( avg > maxMPP)
        {
            maxMPP = avg;
            
            highPw(0,0) = Pw(0,0);
            highPw(1,0) = Pw(0,1);
        }
    }
    cout << "highest Pw found: " << highPw << endl;
    cout << "high accuracy found: " << maxMPP << endl;
}

// this function implements MPP case1,2,3 using optimal
// prior probabiliy on PCA and FLD data
void ClassifyBestPwPCAFLD(Matrix &nXTr)
{
    // using optimal prior probability found from task before
    // which is Pw1 = 0.7 and Pw2 = 0.3
    Matrix Pw(2, 1);
    Pw(0,0) = 0.7;
    Pw(1,0) = 0.3;

    int fea = 22;   // there are 22 features in this data set
    int sumCorrect = 0;
    int sumTotalCount = 0;
    double totalTime = 0.0;
    
    int sampFold = 20;
    int classes = 2;
    int numrow = nXTr.getRow();
    Matrix labelTotal(numrow, 2);
    int labelrow = 0;

    // read in fold data
    Matrix foldData = readData("foldTen.dat", sampFold );
    
    // read one row at a time from the 10 folds and assign the current row read as
    // fold for building validating set. Then assign rest of the current to use as
    // training set using classifier classify the validating set from training set
    for (int j = 0; j < foldData.getRow(); j++ )
    {
        // current fold
        Matrix S = subMatrix(foldData, j,0,j,foldData.getCol()-1);

        // build the validating set from the one fold just read
        Matrix test = getTestingData(S, nXTr);

        // build the training set from leaving the current fold and including the rest of the
        // fold
        Matrix training = getTrainingData(foldData, nXTr, j );

        // to hold class label for testing samples
        Matrix label (test.getRow(), 2);
        
        int foldclass = 2; // there are 2 class category in this data set
        int CorrectCount = 0;
        
        
        // start timming the classification completion for one fold
        // of the validating set
        clock_t start = clock();
        
        
        
        // for each sample in the validating set use the MPP
        // to find the class label of the sample
        for (int i = 0; i < test.getRow(); i++)
        {
            Matrix sample = transpose(subMatrix(test, i,0,i, test.getCol()-2));
            
            // call MPP to perform classification
            label(i,0) = test(i,test.getCol()-1);
            label(i,1) = mpp(training, sample, classes, 2, Pw);
            labelTotal(labelrow, 0) = test(i,test.getCol()-1);
            labelTotal(labelrow, 1) = label(i,1);
            labelrow++;

            if (label(i,1) == test(i,test.getCol()-1))
            {
                CorrectCount++;
            }
        }
        
        sumCorrect = sumCorrect + CorrectCount;
        sumTotalCount = sumTotalCount + test.getRow();
        
        cout << "Correct Rate: " << (((float) CorrectCount) / test.getRow())*100 << endl;
        
        clock_t end = clock();
        totalTime = totalTime + (((double) (end-start)) / 1000000);
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    if (nXTr.getCol() > 2)
    {
        // get the performance metrics for the classification tested
        writeData(labelTotal, "labelTotalNormOptPP.dat");
        
        DerivePerformanceMetric ( labelTotal,4 ); // PCA
    }
    else
    {
        writeData(labelTotal, "labelTotalNormOptPP.dat");
        // get the performance metrics for the classification tested
        DerivePerformanceMetric ( labelTotal,5 ); // FLD
    }
    
    cout << "Average Correct Rate: " << (((float) sumCorrect) / sumTotalCount)*100 << endl;
    cout << "Average Running Time: " << totalTime / 10 << " seconds" << endl;
}

// This method gets the points TPR and FPR for the
// ROC curve drawing
void GetROC(Matrix &nXTr)
{
    Matrix Pw(2, 1);

    int fea = 22;   // there are 22 features in this data set
    int sumCorrect = 0;
    int sumTotalCount = 0;
    double totalTime = 0.0;
    int numrow = nXTr.getRow();
    Matrix labelTotal(numrow, 2);
    int labelrow =0;
    
    int sampFold = 20;
    int classes = 2;
    // read in fold data
    Matrix foldData = readData("foldTen.dat", sampFold );

    // variy the prior probabity over the range 0 to 1 with increment of 0.05
    // and get the performance metric in each case for classification rule MAP applied
    for ( int k = 0; k < 21; k++)
    {
        Pw(0,0) = (float)k/20;
        
        Pw(1,0) = 1 - (Pw(0,0));
        labelTotal.initMatrix(-1);

        labelrow =0;

        // read one row at a time from the 10 folds and assign the current row read as
        // fold for building validating set. Then assign rest of the current to use as
        // training set using classifier classify the validating set from training set
        for (int j = 0; j < foldData.getRow(); j++ )
        {
            // current fold
            Matrix S = subMatrix(foldData, j,0,j,foldData.getCol()-1);

            // build the validating set from the one fold just read
            Matrix test = getTestingData(S, nXTr);

            // build the training set from leaving the current fold and including the rest of the
            // fold
            Matrix training = getTrainingData(foldData, nXTr, j );

            // to hold class label for testing samples
            Matrix label (test.getRow(), 2);
            
            int foldclass = 2; // there are 2 class category in this data set
            int CorrectCount = 0;
            
            // for each sample in the validating set use the MPP
            // to find the class label of the sample
            for (int i = 0; i < test.getRow(); i++)
            {
                Matrix sample = transpose(subMatrix(test, i,0,i, test.getCol()-2));

                // call MPP to perform classification
                label(i,0) = test(i,test.getCol()-1);
                label(i,1) = mpp(training, sample, classes, 3 , Pw);
                labelTotal(labelrow, 0) = label(i,0);
                labelTotal(labelrow, 1) = label(i,1);
                labelrow++;
            }
        }
        
        cout << "PW0 is: " << Pw(0,0) << endl;
        cout << "PW1 is: " << Pw(0,1) << endl;
        DerivePerformanceMetricROC ( labelTotal);
    }
}

// this function calculates the performance metrics for the classification rule MAP used
// on the each type of data set tested for the ROC curve
void DerivePerformanceMetricROC ( Matrix & tested)
{
    double Sensitivity;
    double Specificity;
    double Precision;
    double Recall;
    double Accuracy;
    double TPR;
    double FPR;
    int CorrectCountMPP = 0;
    int TP; // true positive number
    int TN; // true negative number
    int FP; // false positive number
    int FN; // false negative number
    TN = 0;
    TP = 0;
    FP = 0;
    FN = 0;
    Sensitivity = 0.0;
    Specificity = 0.0;
    Precision = 0.0;
    Recall = 0.0;
    Accuracy = 0.0;
    TPR =0.0;
    FPR = 0.0;
    int row = tested.getRow();
    int col = tested.getCol();
    
    for (int i=0; i<row; i++) {
        if (tested(i,0) == tested(i,1))
        {
            CorrectCountMPP++;
            if ( tested(i,0) == 1) // truth yes, observed yes
            {
                TP++;
            }
            else    // truth no, observed no
            {
                TN++;
            }
        }
        else
        {
            if (tested(i,0) == 1) // truth yes, observed no
            {
                FN++;
            }
            else    // truth no, observed yes
            {
                FP++;
            }
        }
    }
    
    Sensitivity = ((double)(TP))/(TP+FN);
    Specificity = ((double)(TN))/(TN+FP);
    Precision = ((double)(TP))/(TP+FP);
    Recall = ((double)(TP))/(TP+FN);
    Accuracy = ((double)((TP+TN)))/(TP+TN+FN+FP);
    FPR = ((double)(FP))/(FP+TN);
    TPR = ((double)(TP))/(TP+FN);
    
    cout << "The FPR rate is " << FPR << endl;

    cout << "The TPR rate is " << TPR << endl;
    
}

// this function performs the KNN classification with different
// k values passed using full Euclidean distance metric
void RunKNN (Matrix &nXTr, int caseNum)
{
    int fea = 22;   // there are 22 features in this data set
    int sumCorrect = 0;
    int sumTotalCount = 0;
    double totalTime = 0.0;
    
    int sampFold = 20;
    int numrow = nXTr.getRow();
    int classes = 2;
    Matrix labelTotal(numrow, 2);
    
    // read in fold data
    Matrix foldData = readData("foldTen.dat", sampFold );
    
    // read one row at a time from the 10 folds and assign the current row read as
    // fold for building validating set. Then assign rest of the current to use as
    // training set using kNN classifier c;assify the validating set from training set
    int labelrow = 0;
    for (int j = 0; j < foldData.getRow(); j++ )
    {
        // current fold
        Matrix S = subMatrix(foldData, j,0,j,foldData.getCol()-1);

        // build the validating set from the one fold just read
        Matrix test = getTestingData(S, nXTr);

        // build the training set from leaving the current fold and including the rest of the
        // fold
        Matrix training = getTrainingData(foldData, nXTr, j );

        // to hold class label for testing samples
        Matrix label (test.getRow(), 2);
        
        
        int foldclass = 2; // there are 2 class category in this data set
        int CorrectCount = 0;
        
        // start timming the kNN classification completion for one fold
        // of the validating set
        clock_t start = clock();
        
        // for each sample in the validating set use the kNN
        // with original Euclidean disable to find the class label of the sample
        for (int i = 0; i < test.getRow(); i++)
        {
            Matrix sample = subMatrix(test, i,0,i, test.getCol()-2);
            
            // call MPP to perform classification
            label(i,0) = test(i,test.getCol()-1);

            label(i,1) = KNNClassifierEuclidian(training, sample, caseNum);

            labelTotal(labelrow, 0) = test(i,test.getCol()-1);
            labelTotal(labelrow, 1) = label(i,1);
            labelrow++;
            
            if (label(i,1) == test(i,test.getCol()-1))
            {
                CorrectCount++;
            }
        }
        
        // get the performance metrics for the classification tested
        
        writeData(labelTotal, "labelKNN_NormData_K_1.dat");
      //  writeData(labelTotal, "labelKNN_PCAData_K_1.dat");
       // writeData(labelTotal, "labelKNN_FLDData_K_1.dat");
        
       // writeData(labelTotal, "labelKNN_NormData_K_5.dat");
     //    writeData(labelTotal, "labelKNN_PCAData_K_5.dat");
       // writeData(labelTotal, "labelKNN_FLDData_K_5.dat");

       // writeData(labelTotal, "labelKNN_NormData_K_10.dat");
        // writeData(labelTotal, "labelKNN_PCAData_K_10.dat");
       // writeData(labelTotal, "labelKNN_FLDData_K_10.dat");

      //  writeData(labelTotal, "labelKNN_NormData_K_15.dat");
        // writeData(labelTotal, "labelKNN_PCAData_K_15.dat");
       // writeData(labelTotal, "labelKNN_FLDData_K_15.dat");

        sumCorrect = sumCorrect + CorrectCount;
        sumTotalCount = sumTotalCount + test.getRow();
        
        cout << "Correct Rate: " << (((float) CorrectCount) / test.getRow())*100 << endl;
        
        clock_t end = clock();
        totalTime = totalTime + (((double) (end-start)) / 1000000);
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
    }
    
    
    
    cout << "Average Correct Rate: " << (((float) sumCorrect) / sumTotalCount)*100 << endl;
    cout << "Average Running Time: " << totalTime / 10 << " seconds" << endl;
    DerivePerformanceMetric ( labelTotal,caseNum );
}

// this function implements the K-means classifier
void RunKmeansClustering(Matrix &XTr)
{
    
    int row;          // for rows in the matrix
    int col;		  // for column in the matrix
    
    // number of clunters center to test with
    // we used only result from cluster = 2
    // other values are used for just testing purpose
    // int numclusters = 128;
    //int numclusters = 64;
   // int numclusters = 32;
    int numclusters = 2;
    row = XTr.getRow();
    col = XTr.getCol()-1;
    
    Matrix centers(numclusters, col);     // for random centroids to start with
    Matrix newcenters(numclusters, col);  // for updated centroids in case of WTA

    Matrix clusterAssign(row, 1);         // assigned clusters to the test samples
    
    
    // time the convergence of the algorithms
    clock_t start = clock();
    
    // get the random cluster centers from the test data corresponponding
    // to the number of cluster tested in the program
    for ( int i=0; i<numclusters; i++)
    {
        int randSamp = (rand ( ) % row ) ;
        
        for (int j=0; j < col; j++)
        {
            centers(i,j) = XTr(randSamp, j);
        }
    }
    
    //
    int done = 0;
    int epoc = 0;
    
        Matrix samp( 1 , col);
        Matrix samp1(1, col);
        clusterAssign.initMatrix(-1);
        
        // stopping creteria when there is not change in the cluster center assigment to all of the test
        // data set
        while ( !done)
        {
            cout << "epoc: " << epoc << endl;
            epoc++;
            done = 1;
            int index;
            
            // for each test sample in the data set compare its Euclidean distance to
            // each of the cluster center and find the cluster center with the minimum
            // distance
            for ( int i = 0; i < XTr.getRow(); i++ )
            {
                double sumDistance = 0.0;
                double min;
                int j;
                for ( int h =0; h < col; h++)
                {
                    samp(0,h) = XTr(i,h);
                }
                
                for ( j =0 ;j < centers.getRow(); j++)
                {
                    for ( int h1 =0; h1 < col; h1++)
                    {
                        samp1(0,h1) = centers(j,h1);
                    }
                    
                    for ( int k =0; k < samp.getCol(); k++)
                    {
                        sumDistance = sumDistance + pow((samp(0,k) - samp1(0,k)), 2);
                    }
                    sumDistance = sqrt(sumDistance);
                    
                    if ( j == 0)
                    {
                        min = sumDistance;
                        index = j;
                    }
                    else
                    {
                        if ( sumDistance < min )
                        {
                            min = sumDistance;
                            //l = j;
                            index = j;
                        }
                    }
                }
                
                // if the found minimum distance cluster center if not the same as what is already
                // assigned cluster center to the test sample then save that cluster as the nearest cluster
                // center for the sample
                if ((int)clusterAssign(i, 0) != index)
                {
                    // set the flag that the change has been made in the cluster assignment
                    done = 0;
                    clusterAssign(i, 0) = index;
                }
            }
            
            // exit the nearest cluster finding loop when there is
            // no change in cluster assignment
            if(done)
            {
                done = 1;
                break;
            }
            
            int val;
            int num =0;
            int col = 3;
            
            Matrix sum(1, col);
            
            // Compute the sample mean of each cluster
            // and that will be the cluster centers to compare with
            // for the reassignment of each sample to the cluster with the nearest mean
            for ( int i = 0; i < numclusters; i++)
            {
                sum.initMatrix(0.0);
                num = 0;
                for ( int j = 0; j < clusterAssign.getRow(); j++ )
                {
                    if ((int)clusterAssign(j,0) == i )
                    {
                        for (int k=0; k< col; k++)
                        {
                            sum(0, k) = sum(0,k) + XTr(j,k);
                            
                        }
                        num++;
                    }
                }
                if(num !=0)
                {
                    for (int k = 0; k < col; k++)
                    {
                        centers(i,k) = sum(0, k) / num;
                    }
                }
            }
        }
        
    
        Matrix FinalData(row,2);
    for ( int i=0; i< row; i++ )
    {
        int r;
        r = clusterAssign(i,0);
        // store class label from assined cluster center
        // for the cunfusion matrix
        FinalData(i,0) = XTr(i, XTr.getCol()-1);
        FinalData(i,1) = XTr(r, XTr.getCol()-1);
    }
    DerivePerformanceMetric( FinalData, 128);
    writeData(FinalData, "LabelKMeansNorm128.dat");
    clock_t end = clock();
    cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
}

// this performs the WTA classification
void RunWTAClustering(Matrix &XTr)
{
    
    int row;          // for rows in the matrix
    int col;		  // for column in the matrix
    
    // number of clunters center to test with
    // int numclusters = 128;
   // int numclusters = 64;
    int numclusters = 2;
    //int numclusters = 32;
    
    
    row = XTr.getRow();
    col = XTr.getCol()-1;
    
    
    
    Matrix centers(numclusters, col);     // for random centroids to start with
    Matrix newcenters(numclusters, col);  // for updated centroids in case of WTA

    Matrix clusterAssign(row, 1);         // assigned clusters to the test samples
    
    
    // time the convergence of the algorithms
    clock_t start = clock();
    
    // get the random cluster centers from the test data corresponponding
    // to the number of cluster tested in the program
    for ( int i=0; i<numclusters; i++)
    {
        int randSamp = (rand ( ) % row ) ;
        
        for (int j=0; j < col; j++)
        {
            centers(i,j) = XTr(randSamp, j);
        }
    }
    
    int done = 0;
    int epoc = 0;
    
            Matrix samp( 1 , col);
        Matrix samp1(1, col);
        double LR = 0.1;
        int g = -1;
        int p = 0;
        int valuetobepushed = -1;
        int s = 0;
        int success = 0;
        Matrix tenResults(1,10);
        clusterAssign.initMatrix(-1);
        
        // if there is no change in the cluster center
        // assignment for 10 epoch although the learning rate has been
        // decreasing then the clustering is all done
        while ( !success)
        {
            epoc++;
            cout << "epoc: " << epoc << endl;
            cout << "learning rate: " << LR << endl;
            done = 1;
            int index;
            
            // for each test sample in the data set compare its Euclidean distance to
            // each of the cluster center and find the cluster center with the minimum
            // distance
            for ( int i = 0; i < XTr.getRow(); i++ )
            {
                double sumDistance = 0.0;
                double min;
                int j;
                for ( int h =0; h < col; h++)
                {
                    samp(0,h) = XTr(i,h);
                }
                
                for ( j =0 ;j < centers.getRow(); j++)
                {
                    for ( int h1 =0; h1 < col; h1++)
                    {
                        samp1(0,h1) = centers(j,h1);
                    }
                    
                    for ( int k =0; k < samp.getCol(); k++)
                    {
                        sumDistance = sumDistance + pow((samp(0,k) - samp1(0,k)), 2);
                    }
                    sumDistance = sqrt(sumDistance);
                    if ( j == 0)
                    {
                        min = sumDistance;
                        index = j;
                    }
                    else
                    {
                        if ( sumDistance < min )
                        {
                            min = sumDistance;
                            
                            index = j;
                        }
                    }
                }
                
                // if the found minimum distance cluster center if not the same as what is already
                // assigned cluster center to the test sample then save that cluster as the nearest cluster
                // center for the sample
                if (((int)clusterAssign(i, 0) != index))
                {
                    
                    g++;
                    // set the flag that the change has been made in the cluster assignment
                    done = 0;
                    clusterAssign(i, 0) = index;
                    
                    // the winner has identified so update the winner cluster center value
                    for ( int l =0; l<col; l++)
                    {
                        newcenters(index,l) = centers(index,l) + (LR * ((XTr(i,l) - centers(index,l))));
                    }
                }
                
            }
            // at the end of the one epoch the found updated winner cluster centers
            // are the cluster centers to be compared with for next epoch
            centers = newcenters;
            
            // decrease the learning rate to guarantee the convergence of the algorithm
            LR = 0.9 * LR;
            
            // keep track of the ten consecutive epoch for which the
            // cluster assignment has not changed even though the
            // learnig rate has been decreasing
            if ( done != 0)
            {
                valuetobepushed=1;
                
            }
            else {
                valuetobepushed=0;
            }
            
            for (int x=0; x <9; x++){
                tenResults(0,x) = tenResults(0,x+1);
            }
            tenResults(0,9) = valuetobepushed;
            s =0;
            for ( int h =0; h < 10; h++)
            {
                s += tenResults(0,h);
            }
            
            
            if ( s==10){
                success = 1;
            }
            
        }
        Matrix FinalData(row,2);

        for ( int i=0; i< row; i++ )
        {
            int r;
            r = clusterAssign(i,0);
            FinalData(i,0) = XTr(i, XTr.getCol()-1);
            FinalData(i,1) = XTr(r, XTr.getCol()-1);;
        }
        DerivePerformanceMetric( FinalData, 1111128);
      //  writeData(FinalData, "LabelWTANorm128.dat");
        clock_t end = clock();
        cout << "Running Time: " << (double) (end-start)/ 1000000 << " seconds" << endl;
}

// this performs the classifier fusion using
// Naive Bayes
void RunNaiveBayes()
{
   // Matrix x1 = readData("ConfusionMatrix4", 2);
    Matrix x1 = readData("ConfusionMatrixKNNPCA1", 2);
    Matrix x2 = readData("ConfusionMatrixKmeans2", 2);
    Matrix M(2,4);
    
    double n1 = x1(0,0);
    double n2 = x1(0,1);
    double n3 = x1(1,0);
    double n4 = x1(1,1);
    
    double m1 = x2(0,0);
    double m2 = x2(0,1);
    double m3 = x2(1,0);
    double m4 = x2(1,1);
    
    // get the probability
    x1(0,0) = ((double)n1)/ (n1+n2);
    x1(0,1) = ((double)n2)/ (n1+n2);
    x1(1,0) = ((double)n3)/ (n3+n4);
    x1(1,1) = ((double)n4)/ (n3+n4);
    
    x2(0,0) = m1/ (m1+m2);
    x2(0,1) = m2/ (m1+m2);
    x2(1,0) = m3/ (m3+m4);
    x2(1,1) = m4/ (m3+m4);

    int j;
    
    // do the multiplication part
    for (int i=0; i<2; i++)
    {
      for (j=0; j<2; j++)
      {
          M(i,j) = x1(i,0) * x2(i,j);
          
      }
        for(int k = 0; k <2; k++ )
        {
            M(i,j) = x1(i,1) * x2(i,j);
            j++;
        }
    }
    
   // Matrix y1 = readData("labelTotalPCAOptPP.dat", 2);
    Matrix y1 = readData("labelKNN_PCAData_K_1.dat", 2);
    Matrix y2 = readData("NewLabelKmeanNorm2", 2);

    
    int row = y1.getRow();
    Matrix label (row, 2);
    
    int l1,l2;
    
    // assign the label based on label with highest
    // probability
    for(int i =0; i < row; i++)
    {
        l1 = y1(i,1);
        l2 = y2(i,1);
        
        label(i,0) = y1(i,0);
        
        if (l1 == 1 && l2 == 1)
        {
           if( M(0,0) > M(1,0))
           {
               label(i,1) = 1;
           }
            else
            {
                label(i,1) = 0;
            }
        }
        else if (l1 == 1 && l2 == 0)
        {
            if( M(0,1) > M(1,1))
            {
                label(i,1) = 1;
            }
            else
            {
                label(i,1) = 0;
            }
        }
        else if (l1 == 0 && l2 == 1)
        {
            if( M(0,2) > M(1,2))
            {
                label(i,1) = 1;
            }
            else
            {
                label(i,1) = 0;
            }
        }
        else if (l1 == 0 && l2 == 0)
        {
            if( M(0,3) > M(1,3))
            {
                label(i,1) = 1;
            }
            else
            {
                label(i,1) = 0;
            }
        }
    }
    DerivePerformanceMetric(label, 1);
}

// this is used to make the labels from K-mean
// corresponding to classifier using K-fold data
void RunClusterLabel()
{
     Matrix foldData = readData("foldTen.dat", 20 );
    int k =0;
     Matrix lbKM = readData("LabelKMeansNorm2.dat", 2 );
    Matrix newLb(lbKM.getRow(), 2);
    int num;
    for (int i =0; i< foldData.getRow(); i++)
    {
        for (int j=0; j< foldData.getCol(); j++)
        {
            num = foldData(i,j);
            if (num != -1)
            {
                newLb(num,0) = lbKM(k,0);
                newLb(num,1) = lbKM(k,1);
                k++;
            }
        }
    }
    writeData(newLb, "NewLabelKmeanNorm2");
}


