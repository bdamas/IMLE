
template<class Learner>
void cos1D_train( Learner &imleObj)
{
    // Training dataset size
    const int NPOINTS_TRAIN = 2000;
    // Noise model
    const Scal NOISE_Z = 0.02;
    const Scal NOISE_X = 0.1;
    // Random number generation
    boost::mt19937 rng;
    boost::uniform_01<Scal> uniform;
    boost::normal_distribution<Scal> normal;
    // Input and output Eigen vectors
    Vec z(1), x(1);

    // Training
    cout << "\t --- Training, y = cos(x), " << NPOINTS_TRAIN << " training points ---" << endl;
    z[0] = 0.0;
    for(int k = 0; k < NPOINTS_TRAIN; k++)
    {
        z[0] += uniform(rng) * NOISE_Z;
        x[0] = cos(z[0]) + NOISE_X * normal(rng);

        imleObj.update(z,x);

        if( z[0] > 4*M_PI )
            z[0] = 0.0;
        else
            z[0] += 0.1;
    }
}


template<class Learner>
void cos1D_display( Learner &imleObj)
{
    // Number of Experts
    cout << "\t --- Number of experts: ---" << endl;
    cout << imleObj.getNumberOfExperts() << endl;

    // Common Sigma
    cout << "\t --- Common Sigma: ---" << endl;
    cout << imleObj.getSigma() << endl;

    // Psi
    cout << "\t --- Common Psi: ---" << endl;
    cout << imleObj.getPsi() << endl;

    // Showing internal data
    cout << "\t --- Internal Data: ---" << endl;
    cout << imleObj << endl;
}

template<class Learner>
void cos1D_predict( Learner &imleObj)
{
    // Random number generation
    boost::mt19937 rng;
    boost::uniform_01<Scal> uniform;
    // Query dataset size
    const int NPOINTS_QUERY = 10;
    // Input and output Eigen vectors
    Vec z(1), x(1);

    // Forward Prediction
    cout << "\t --- Forward Prediction (random input locations): ---" << endl;
    for(int k = 0; k < NPOINTS_QUERY; k++)
    {
        z[0] = uniform(rng) * 4*M_PI;

        x = imleObj.predict(z);
        // Alternatively:
        //      imleObj.predict();
        //      x = getPrediction()[0];

        cout << "\t cos(" << z[0] << ") = " << cos(z[0]) << endl;
        cout << "\tpred(" << z[0] << ") = " << x[0] << "  (error = " << cos(z[0])-x[0] << ")" << endl;
        cout << "\t\t    Jacobian(" << z[0] << ") = " << -sin(z[0]) << endl;
        cout << "\t\tpredJacobian(" << z[0] << ") = " << imleObj.getPredictionJacobian()(0,0) << endl;
    }

    // Inverse Prediction
    cout << "\t --- Inverse Prediction (random output locations): ---" << endl;
    for(int k = 0; k < NPOINTS_QUERY; k++)
    {
        x[0] = uniform(rng) * 2.0 - 1.0;
        Scal inv_x = acos(x[0]);

        imleObj.predictInverse(x);

        vector<Scal> invSol;
        int nSol = imleObj.getNumberOfInverseSolutionsFound();
        for( int k = 0; k < nSol; k++ )
            invSol.push_back( imleObj.getInversePredictions()[k][0] );
        sort(invSol.begin(), invSol.end() );

        cout << "\tinvCos(" << x[0] << ") = ( " << inv_x << ", " << 2.0*M_PI-inv_x << ", " << 2.0*M_PI+inv_x << ", " << 4.0*M_PI-inv_x << " )" << endl;
        cout << "\t  pred(" << x[0] << ") = ( ";
        for( int k = 0; k < nSol; k++ )
            cout << invSol[k] << ", ";
        cout << ")" << endl;
    }
}

