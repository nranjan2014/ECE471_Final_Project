// Source code: C++ code for Majority Voting Fusion:

int *fusion(int *vote1, int* vote2, int* vote3)
{
    int votes, i; 
    int *fused;
    fused = new int[195];

    for(i=0; i<195; i++)
    {
        votes = 0; 
        votes += vote1[i] + vote2[i] + vote3[i]; 
        if(votes >= 2)
            fused[i] = 1; 
        else
            fused[i] = 0;
    }
    return fused; 
}

