#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>

#include "Matrix.h"
#include "Pr.h"

using namespace std;

#define Usage "Usage: ./ranjan_proj1 synth.tr synth.te classes number_of_feature cases \n"
//mpp(tr, sample, classes, cases, Pw);
int mpp(Matrix &Tr, Matrix &TeSp, int cls, int cas, Matrix &Pw );

int main ( int argc, char *argv[])
{
    int numfea;   // number of features
    int classes;  // number of classes
    int cases;    // number of cases used for MPP
    int numcol;   // number of columns in the data set

    Matrix tr, te, Typetr0, Typetr1, Typete0, Typete1;

    // check for the correct number of arguments

    if (argc < 6)
    {
        cout << Usage;
        exit(1);
    }

    // read in the number of features in the data set

    numfea = atoi(argv[4]);
    classes = atoi(argv[3]);
    cases = atoi(argv[5]);

    numcol = numfea+1;

    // read in traning data set from synth.tr file and save in Matrix tr
    // Theere are three columns in this training set first two are the features     
    // of the data set and the last one indicates the category of the samples.

    tr = readData(argv[1], numcol);

    // for testing code works so far to be removed after final draft of the code

    // output whole traning data matrix for testing
    cout << "\nOutput the whole matrix for testing...\n";
    cout << tr;

    // now read in testing data set from synth.te file in Matrix tr
    te = readData(argv[2], numcol);

    // output whole testing data matrix for testing
    cout << "\nOutput the whole matrix for testing...\n";
    cout << te;

    // there are two types of samples in the traning and the testing data set 
    // which is type 0 and type 1
    //get the samples that belong to type 0 and save it to correspondingly 
    // named Matrices

    // firts get the two types of samples from traning data set and save it to
    // Matrix Typetr0, Typetr1

    Typetr0 = getType(tr, 0);
    Typetr1 = getType(tr, 1);


    // then get the two types of samples from testing data set and save it to
    // Matrix Typete0, Typete1

    Typete0 = getType(te, 0);
    Typete1 = getType(te, 1);

     
    // for testing code works so far to be removed after final draft of the code
    // write these type matrices to corresponding files

    writeData(Typetr0, "Typetr0.dat");
    writeData(Typetr1, "Typetr1.dat");
    writeData(Typete0, "Typete0.dat");
    writeData(Typete1, "Typete1.dat");
 
    // get the number of rows for training data set

    int numrowTr = tr.getRow();

    // get the number of rows for testing data set

    int numrowTe = te.getRow();

    // prepare the labels and error count
    Matrix labelMPP(numrowTe, 1);

    // calculate error rate for MPP
    int errCountMPP = 0; 

    // we assume here equal prior probability
    // assign prior probability
    Matrix Pw(classes, 1);

    for (int i =0; i<classes; i++)
    {
        Pw(i,0) = (float) 1/classes;
    }

    Matrix meanTr0, meanTr1, covarTr0, covarTr1;

    meanTr0 = mean(Typetr0, numfea);
    meanTr1 = mean(Typetr1, numfea);

    covarTr0 = cov(Typetr0, numfea);
    covarTr1 = cov(Typetr1, numfea);

    // for testing 
    cout << "mean class 0: " << meanTr0 << endl;
    cout << "mean class 1: " << meanTr1 << endl;
    cout << "covariance class 0: " << covarTr0 << endl;
    cout << "covariance class 1: " << covarTr1 << endl;

    Matrix diff, inverCovar;
    double det;
/*
    for ( int i =0; i < numrowTe; i++)
    {
       Matrix eachSample;

       eachRow = subMatrix(tr, i, 0, i, 1);
       
       diff = transpose(eachRow - );


    } */
    
/*
    // perform classification
    for (int i=0; i < numrowTe; i++)
    {
      // classify one test sample at a time, get one sample from the test data set
      Matrix sample = transpose(subMatrix(te, i, 0, i, numfea-1));
   
      // call MPP to perform classification
      labelMPP(i,0) = mpp(tr, sample, classes, cases, Pw);

      // check if the classification result is correct or not
      if (labelMPP(i,0) != te(i,numfea))
      {
         errorCountMPP++;
      }
    }
*/
    // calculate accuracy
    cout << "The error rate using MPP is = " << (float) errCountMPP/numrowTe << endl;
    return 0;

}

//int mpp( Matrix &trDt, Matrix 
