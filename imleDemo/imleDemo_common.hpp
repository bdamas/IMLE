
template<class Learner>
void cos1D_train( Learner &imleObj, int nTrain = 2000)
{
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
    cout << "\t --- Training, y = cos(x), " << nTrain << " training points ---" << endl;
    z[0] = 0.0;
    for(int k = 0; k < nTrain; k++)
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

template<class Learner>
void cos1D_display( Learner &imleObj)
{
    // Dimensions:
    cout << "Input dimension: "  << imleObj.inputDim() << endl;
    cout << "Output dimension: " << imleObj.outputDim() << endl;

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
void cos1D_testSaveLoadAndParamHandling( Learner &imleObj )
{
	// Parameter handling
	typename Learner::Param param = imleObj.getParameters();
	cout << "\t-- Initial parameters: --" << endl;
	imleObj.displayParameters();

	imleObj.loadParameters("cos1D.xml");
	cout << "\t-- Now the IMLE object has default parameters: --" << endl;
	imleObj.displayParameters();

	imleObj.setParameters(param);
	cout << "\t-- Reverting to initial parameters: --" << endl;
	imleObj.displayParameters();

	// Training (10 points)
    cos1D_train(imleObj, 10);

    // Display info
    cos1D_display(imleObj);

    // Save
    imleObj.save("cos1D_demo.imle");

    // Reset
    imleObj.reset();
    cout << imleObj;

    // Train again
    cos1D_train(imleObj, 20);
    cos1D_display(imleObj);

    // Load
    imleObj.load("cos1D_demo.imle");
    cout << imleObj;
}

template<class Learner>
void planar_robot_training( Learner &imleObj, int nTrain = 10000 )
{
    // Noise model
    const Scal NOISE_Z = 0.02;
    const Scal NOISE_X = 0.1;
    // Random number generation
    boost::mt19937 rng;
    boost::uniform_01<Scal> uniform;
    boost::normal_distribution<Scal> normal;
    // Input and output Eigen vectors
    Vec z(2), x(2);

    // Training
    cout << "\t --- Training, y = cos(x), " << nTrain << " training points ---" << endl;
    z[0] = 0.0;
    for(int k = 0; k < nTrain; k++)
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
void doubleCos1D_train( Learner &imleObj, int nTrain = 2000)
{
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
	cout << "\t --- Training, y = {cos(x), cos(x)+4}, " << nTrain << " training points ---" << endl;
	Scal zTrain = 0.0;
    for(int k = 0; k < nTrain; k++)
    {
        zTrain += uniform(rng) * NOISE_Z;

        if( zTrain > 0.0 )      // Lower branch
        {
            z[0] = zTrain;
            x[0] = cos(z[0]) + NOISE_X * normal(rng);
        }
        else                   // Upper branch
        {
            z[0] = zTrain + 4*M_PI;
            x[0] = cos(z[0]) + 4.0 + NOISE_X * normal(rng);
        }

        imleObj.update(z,x);

		if( zTrain > 4*M_PI )
            zTrain = -4*M_PI;
        else
            zTrain += 0.1;
    }
}

template<class Learner>
void doubleCos1D_eval( Learner &imleObj)
{
	const int nTest = 100;
    typename Learner::ArrayX x;
	Vec zTest = Vec::LinSpaced(100, 0.0, 4*M_PI);
	int nSol;
	Scal err;

	Scal sumSol = 0;
	Scal sumErrSqr = 0.0;

	for( int i = 0; i < nTest; i++)
	{
		imleObj.predictMultiple(zTest.row(i));

		nSol = imleObj.getNumberOfSolutionsFound();
		x = imleObj.getMultiplePredictions();

        Scal x1 = cos(zTest[i]);
        Scal x2 = x1 + 4.0;

        cout << "z = " << zTest[i] << " --> \tTargets: " << x1 << "   " << x2 << "\n\t\t    Predictions: ";
		for(int k = 0; k < nSol; k++)
		{
		    cout << x[k][0] << "   ";

			if( x[k][0] < x1 + 2.0 )	// Getting the closest true value to the IMLE estimate, and using it to calculate the error
				err = x[k][0] - x1;
			else
				err = x[k][0] - x2;

			sumErrSqr += err*err;
		}
		sumSol += nSol;
        cout << endl;
	}
	cout << "\t --- doubleCos1D_eval: ---" << endl;
	cout << "\t\tNumber of experts created: " << imleObj.getNumberOfExperts() << endl;
	cout << "\t\tAverage number of predicted solutions: " << sumSol/nTest << endl;
	cout << "\t\tRMSE: " << sqrt( sumErrSqr / sumSol ) << endl;
}
