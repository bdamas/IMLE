//#include "imle.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

#include <boost/math/distributions/chi_squared.hpp>
#include <Eigen/LU>

#define EXPAND(var) #var "= " << var
#define EXPAND_N(var) #var "=\n" << var

#define IMLE_CRTD		"IMLE object created."
#define IMLE_FNSHD		"IMLE object finished."
#define STR_BFR_UPDT	"You must start IMLE before updating."
#define OPENERR			"IMLE: Could not open file "
#define USNG_DEF_PRM    "IMLE: Using default parameters."

/*
 * Static Members
 */
IMLE_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
void IMLE_base::createDefaultParamFile(int _d, int _D, std::string const &fname)
{
	Param param(_d,_D);
#else
void IMLE_base::createDefaultParamFile(std::string const &fname)
{
    Param param;
#endif

    std::ofstream ofs(fname.c_str());
	if( !ofs.is_open() )
	{
		std::cerr <<  fname << " not opened!" << std::endl;
		return;
	}

    boost::archive::xml_oarchive oa(ofs);
    oa & BOOST_SERIALIZATION_NVP(param);
    std::cout << "Created the following configuration file:\n\n";
    param.display();
//    std::cout << param;

    ofs.close();
}


/*
 * IMLE Parameters
 */
IMLE_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
IMLE_base::Param::Param(int _d, int _D)
: d(_d), D(_D)
#else
IMLE_base::Param::Param()
#endif
{
    accelerated = false;
    alpha = 0.999;

    Psi0 = IMLE_base::X::Ones(D);
    sigma0 = 1.0;
    wsigma = pow(2.0,d);
    wpsi = 2*D;

    wNu = 0.0;
    wLambda = 0.1;
    wSigma = pow(2.0,d);
    wPsi = pow(2.0,d);

    nOutliers = 1;
    p0 = 0.1;

    multiValuedSignificance = 0.99;
//	    predictWithOutlierModel = true;
    nSolMin = 1;
    nSolMax = MAX_NUMBER_OF_SOLUTIONS;
    iterMax = 2;

    saveOnExit = false;
    defaultSave = DEFAULT_SAVE;
}


/*
 * Constructors and destructors
 */
#ifdef IMLE_NO_TEMPLATES
IMLE_CLASS_TEMPLATE
IMLE_base::IMLE(int d, int D, int pre_alloc)
: d(d), D(D), param(d,D)
{
    init( pre_alloc );
    reset();
}
#endif

IMLE_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
IMLE_base::IMLE(int d, int D, Param const &prm, int pre_alloc)
: d(d), D(D), param(d,D)
#else
IMLE_base::IMLE(Param const &prm, int pre_alloc)
#endif
{
    init( pre_alloc );
    reset(prm);
}

IMLE_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
IMLE_base::IMLE(int d, int D, std::string const &filename, int pre_alloc)
: d(d), D(D), param(d,D)
#else
IMLE_base::IMLE(std::string const &filename, int pre_alloc)
#endif
{
    init( pre_alloc );
    reset();

	if( !load(filename) )
	    message(USNG_DEF_PRM);
    else
        message("Loaded " + filename);
}

IMLE_CLASS_TEMPLATE
IMLE_base::~IMLE()
{
    if( param.saveOnExit )
        save(param.defaultSave);

	message(IMLE_FNSHD);
}

/*
 * Initialization
 */

IMLE_CLASS_TEMPLATE
void IMLE_base::init(int pre_alloc)
{
    zeroZZ = ZZ::Zero(d,d); //Template: ZZ::Zero()
    infinityZZ = ZZ::Constant( d,d, std::numeric_limits<Scal>::infinity() ); //Template: ZZ::Constant( std::numeric_limits<Scal>::infinity() )
    zeroZ = Z::Zero(d);     //Template: Z::Zero()
    zeroX = X::Zero(D);     //Template: X::Zero()
	infinityX = X::Constant( D, std::numeric_limits<Scal>::infinity() ); //Template: X::Constant( std::numeric_limits<Scal>::infinity() )
    zeroXZ = XZ::Zero(D,d); //Template: XZ::Zero()
    infinityXZ = XZ::Constant( D,d, std::numeric_limits<Scal>::infinity() ); //Template: XZ::Constant( std::numeric_limits<Scal>::infinity() )

    sNearest.reserve(INIT_SIZE);
    sNearestInv.reserve(INIT_SIZE);

    experts.reserve( pre_alloc );

	predictions.reserve(MAX_NUMBER_OF_SOLUTIONS);
	predictionsVar.reserve(MAX_NUMBER_OF_SOLUTIONS);
	predictionsWeight.reserve(MAX_NUMBER_OF_SOLUTIONS);
	predictionsJacobian.reserve(MAX_NUMBER_OF_SOLUTIONS);
	predictionsVarJacobian.reserve(MAX_NUMBER_OF_SOLUTIONS);
	invPredictions.reserve(MAX_NUMBER_OF_SOLUTIONS);
	invPredictionsVar.reserve(MAX_NUMBER_OF_SOLUTIONS);
	invPredictionsWeight.reserve(MAX_NUMBER_OF_SOLUTIONS);

	message(IMLE_CRTD);
}

IMLE_CLASS_TEMPLATE
void IMLE_base::reset()
{
    experts.clear();
    set_internal_parameters();

    noise_to_go = 0;
}

IMLE_CLASS_TEMPLATE
void IMLE_base::reset(Param const &prm)
{
    experts.clear();
    setParameters(prm);

    noise_to_go = 0;
}

IMLE_CLASS_TEMPLATE
void IMLE_base::setParameters(Param const &prm)
{
#ifdef IMLE_NO_TEMPLATES
    // Check dimensions
    if( prm.d != d || prm.D != D)
    {
        message("IMLE::setParameters: parameter dimensions do not match!!");
        return;
    }
#endif

    param = prm;
    set_internal_parameters();
}

IMLE_CLASS_TEMPLATE
bool IMLE_base::loadParameters(std::string const &fname)
{
#ifdef IMLE_NO_TEMPLATES
    Param newParams(d,D);
#else
    Param newParams;
#endif

    std::ifstream ifs(fname.c_str());
    	if( !ifs.is_open() )
	{
		message(OPENERR + fname + "!!");
		return false;
	}

    assert(ifs.good());
    boost::archive::xml_iarchive ia(ifs);
    ia & BOOST_SERIALIZATION_NVP(newParams);
    ifs.close();

    setParameters(newParams);

	return true;
}

IMLE_CLASS_TEMPLATE
void IMLE_base::set_internal_parameters()
{
    // Number of experts
    M = experts.size();

    // Initial guess for hyperparameters
    if( M == 0 )
    {
        Sigma = Z::Constant(d,param.sigma0);  //Template: Z::Constant(param.sigma0)
        Psi = param.Psi0;
    }

    // Significance test level
    sig_level_noiseX_rbf = exp(-0.5* quantile(boost::math::chi_squared(D), 1.0 - param.p0) );
    sig_level_noiseZ_rbf = exp(-0.5* quantile(boost::math::chi_squared(d), 1.0 - param.p0) );
    sig_level_noiseZX_rbf = exp(-0.5* quantile(boost::math::chi_squared(D+d), 1.0 - param.p0) );

    pNoiseModelX = sig_level_noiseX_rbf / sqrt(Psi.prod());
    pNoiseModelZ = sig_level_noiseZ_rbf / sqrt(Sigma.prod());
    pNoiseModelZX = sig_level_noiseZX_rbf/ sqrt(Psi.prod() * Sigma.prod());

//    sig_level_multi_test = quantile(boost::math::chi_squared(D*(D+1)/2), param.multiValuedSignificance);
}


/*
 * Save and load
 */
IMLE_CLASS_TEMPLATE
bool IMLE_base::save(std::string const &filename)
{
    std::ofstream fs(filename.c_str());

	if( !fs.is_open() )
	{
		message(OPENERR + filename);
		return false;
	}
    boost::archive::text_oarchive archive(fs);

	archive & (*this);
    fs.close();

    return true;
}

IMLE_CLASS_TEMPLATE
bool IMLE_base::load(std::string const &filename)
{
    std::ifstream fs(filename.c_str());
	if( !fs.is_open() )
	{
		message(OPENERR + filename);
		return false;
	}
    boost::archive::text_iarchive archive(fs);

    reset();
	archive & (*this);
    fs.close();

    return true;
}

IMLE_CLASS_TEMPLATE
template<class Archive>
void IMLE_base::serialize(Archive & ar, const unsigned int version)
{
	// Parameters
	ar & param;

    // Experts
    ar & experts;

    // Common Priors
    ar & Sigma;
    ar & Psi;

    // Remaining parameters (only for loading)
    set_internal_parameters();
    ar & noise_to_go;
}


/*
 * UPDATE
 */

IMLE_CLASS_TEMPLATE
void IMLE_base::update(Z const &z, X const &x)
{
#ifdef IMLE_NO_TEMPLATES
    // Check dimensions
    if( z.size() != d || x.size() != D)
    {
        message("IMLE::update: sample point dimensions do not match!!");
        return;
    }
#endif

    // Query experts
    Scal sum_zx = 0.0;
    for(IMLE_TYPENAME Experts::iterator it=experts.begin(); it < experts.end(); it++)
        sum_zx += it->queryZXandH(z,x);

    // Create new expert?
    if( sum_zx < pNoiseModelZX / M || M == 0 )
	{
	    if( noise_to_go > 0)
	    {
	        noise_to_go--;
            return;
	    }

        // Create new linear expert
#ifdef IMLE_NO_TEMPLATES
        experts.push_back( Expert(d,D,z,x,this) );
#else
        experts.push_back( Expert(z,x,this) );
#endif
        experts.back().queryH(z,x);
        M = experts.size();
	}

    EM(z,x);

	noise_to_go = param.nOutliers;
}

IMLE_CLASS_TEMPLATE
void IMLE_base::EM(Z const &z, X const &x)
{
    // Expectation Step
    Scal h, sum_h = 0.0;
    for(IMLE_TYPENAME Experts::iterator it=experts.begin(); it < experts.end(); it++)
        sum_h += it->get_h();

    if( sum_h == 0.0)
    {
        // Create new linear expert
#ifdef IMLE_NO_TEMPLATES
        experts.push_back( Expert(d,D,z,x,this) );
#else
        experts.push_back( Expert(z,x,this) );
#endif
        experts.back().e_step(z,x,1.0);
        experts.back().m_step();
        M = experts.size();
        std::cout << "e_step: sum_h = 0.0! (Should not happen)" << std::endl;
    }
    else if( param.accelerated )
        for(IMLE_TYPENAME Experts::iterator it=experts.begin(); it < experts.end(); it++)
        {
            if( (h = it->get_h()/sum_h) > 0.001 )
            {
                it->e_step( z, x, h );
                it->m_step();
            }

        }
    else
        for(IMLE_TYPENAME Experts::iterator it=experts.begin(); it < experts.end(); it++)
        {
            it->e_step( z, x, it->get_h()/sum_h );
            it->m_step();
        }

    // Maximization Step
    //
    Z sum_diagSigma = zeroZ;
    X sum_diagPsi = zeroX;
    for(IMLE_TYPENAME Experts::iterator it=experts.begin(); it < experts.end(); it++)
    {
        sum_diagSigma += it->invSigma.diagonal();
        sum_diagPsi += it->invPsi;
    }

    // Update common inverse-Wishart prior on Sigma
    Scal tmp = (M*param.wSigma - param.wsigma)/2.0 - 1.0;
    Sigma = param.wSigma * param.wsigma * param.sigma0 * sum_diagSigma;
    Sigma.array() += tmp*tmp;
    Sigma.array() = (Sigma.array().sqrt() + tmp) / (param.wSigma * sum_diagSigma.array());

    // Update common inverse-gamma prior on Psi
    tmp = (M*param.wPsi - param.wpsi)/2.0 - 1.0;
    Psi.array() = (param.wPsi * param.wpsi) * param.Psi0.array() * sum_diagPsi.array();
    Psi.array() += tmp*tmp;
    Psi.array() = (Psi.array().sqrt() + tmp) / (param.wPsi * sum_diagPsi.array());

    // Update outlier model
    pNoiseModelX = sig_level_noiseX_rbf / sqrt(Psi.prod());
    pNoiseModelZ = sig_level_noiseZ_rbf / sqrt(Sigma.prod());
    pNoiseModelZX = sig_level_noiseZX_rbf / sqrt(Psi.prod() * Sigma.prod());
}

/*
 * PREDICTION
 */
IMLE_CLASS_TEMPLATE
IMLE_TYPENAME IMLE_base::X const &IMLE_base::predict(Z const &z)
{

    // Clear predict data structure
    predictions.clear();
    predictionsVar.clear();
    predictionsWeight.clear();

#ifdef IMLE_NO_TEMPLATES
    // Check dimensions
    if( z.size() != d )
    {
        message("IMLE::predict: query dimension does not match input dimension!!");

        sum_p_z.setZero(1);
    }
    else
#endif
    {
        Scal sum_p = 0.0;
        for(int j = 0; j < M; j++)
            sum_p += experts[j].queryZ(z);
        sum_p_z.setConstant(1, sum_p);
        sumW.setConstant(1, sum_p + pNoiseModelZ);
        sumAll = sumW(0);
    }

    if(sum_p_z(0) == 0.0)  // This happens when a prediction is sought too far from the current mixture
    {
        varPhiAuxj.setConstant(M,pNoiseModelZ);
        fInvRj.setZero(D,M);
        invRxj.setZero(D,M);

        predictions.push_back(zeroX);
        predictionsVar.push_back(infinityX);
        predictionsWeight.push_back(0.0);
    }
    else
    {
        varPhiAuxj.resize(M);
        fInvRj.resize(D,M);
        invRxj.resize(D,M);
        for(int j = 0; j < M; j++)
        {
            varPhiAuxj(j) = experts[j].getGamma() * experts[j].get_p_z() + sumW(0);
            fInvRj.col(j) = experts[j].getPredXInvVar() * (experts[j].get_p_z() / varPhiAuxj(j));
            invRxj.col(j) = fInvRj.col(j).asDiagonal() * experts[j].getPredX();
        }

        // Update storage results
        X sumInvRj = fInvRj.rowwise().sum();
        X sumInvRxj = invRxj.rowwise().sum();

        predictions.push_back( sumInvRxj.cwiseQuotient(sumInvRj) );
        predictionsVar.push_back( sumInvRj.cwiseInverse() );
        predictionsWeight.push_back( sum_p_z(0) / sumAll );
    }

    nSolFound = 1;
    sNearest.clear();
    sNearest.resize(M,0);

    solIdx = 0;

    zQuery = z;
    hasPredJacobian = false;
    hasPredErrorReductionDrvt = false;

    return predictions[solIdx];
}



IMLE_CLASS_TEMPLATE
IMLE_TYPENAME IMLE_base::X const &IMLE_base::predictStrongest(Z const &z)
{
    predictMultiple(z);

    Scal w = 0.0;
    int best = 0;
    for(int i=0; i < nSolFound; i++)
        if( predictionsWeight[i] > w )
        {
            w = predictionsWeight[i];
            best = i;
        }

    solIdx = best;

    //          Not needed, already in predict()
    // hasPredJacobian = false;
    // hasPredErrorReductionDrvt = false;

    return predictions[solIdx];
}

IMLE_CLASS_TEMPLATE
void IMLE_base::predictMultiple(Z const &z)
{
    predict(z);

    if( sum_p_z(0) == 0.0 )
        return;

    int newSol1, newSol2, worseSol;
    if( !validForwardSolutions(newSol1, newSol2, worseSol) )
    {

        do
        {
            nSolFound++;
            clusterForwardSolutions(newSol1, newSol2, worseSol);
        } while ( !validForwardSolutions(newSol1, newSol2, worseSol) );

        getForwardSolutions();
    }

    //          Not needed, already in predict()
    // hasPredJacobian = false;
    // hasPredErrorReductionDrvt = false;
}

IMLE_CLASS_TEMPLATE
bool IMLE_base::validForwardSolutions(int &newSol1, int &newSol2, int &worseSol)
{
    X diffX;
    Vec dist(M);
    Vec T = Vec::Zero(nSolFound);
    Vec sumWsq = Vec::Zero(nSolFound);
    for(int j = 0; j < M; j++)
    {
        diffX = experts[j].getPredX() - predictions[sNearest[j]];
        dist(j) = diffX.dot(fInvRj.col(j).asDiagonal() * diffX);

        T[sNearest[j]] += dist(j);
        sumWsq(sNearest[j]) += experts[j].get_p_z()*experts[j].get_p_z();
    }

    worseSol = -1;
    Scal max_p_value = -1.0, p_value;

    for(int k = 0; k < nSolFound; k++)
    {
        Scal dof = (sum_p_z(k) * sum_p_z(k) / sumWsq(k) - 1.0) * D;

        // Small hack to prevent numerical problems
        dof += 1.0;
        if( !(dof >= 1.0) || T(k) <= 0.0)  // The negation is because of NaN problems
            continue;
//            dof = 1.0;

        if( ( p_value = cdf(boost::math::chi_squared(dof), T(k)) ) >= max_p_value )
        {
            max_p_value = p_value;
            worseSol = k;
        }
    }

    if( max_p_value <= param.multiValuedSignificance )
        return true;
    else
    {
        // Find the largest expert deviation in worse solution
        Scal dMax = -1.0;
        for(int j = 0; j < M; j++)
            if( (sNearest[j] == worseSol) && (dist(j) > dMax) )
            {
                dMax = dist(j);
                newSol1 = j;
            }

        // Find the largest expert deviation to newSol1
        dMax = -1.0;
        Scal dist2;
        for(int j = 0; j < M; j++)
            if( sNearest[j] == worseSol )
            {
                diffX = experts[j].getPredX() - experts[newSol1].getPredX();
                dist2 = diffX.dot(fInvRj.col(j).asDiagonal() * diffX);
                if( dist2 > dMax )
                {
                    dMax = dist2;
                    newSol2 = j;
                }
            }

        return false;
    }
}

IMLE_CLASS_TEMPLATE
void IMLE_base::clusterForwardSolutions(int newSol1, int newSol2, int worseSol)
{
    Mat p(M,nSolFound);
    Mat sol(D,nSolFound);

    // "Educated" guesses for the nSolFound solutions
    for(int k = 0; k < nSolFound-1; k++)
        sol.col(k) = predictions[k];
    // Split worse solution in two
    sol.col(worseSol)    = experts[newSol1].getPredX();
    sol.col(nSolFound-1) = experts[newSol2].getPredX();

    for( int nIter = 0; nIter < param.iterMax; nIter++ )
    {
        // E-Step
        for( int j = 0; j < M; j++ )
        {
            for( int k = 0; k < nSolFound; k++ )
            {
                X dist = experts[j].getPredX() - sol.col(k);
                p(j,k) = exp(-0.5*dist.dot(fInvRj.col(j).asDiagonal()*dist));
            }
            Scal pSum;
            if( (pSum = p.row(j).sum()) > 0.0 )
                p.row(j) /= pSum;
        }
        // M-Step
        sol = (invRxj * p).cwiseQuotient( fInvRj * p );
        /* Here I can easily implement k-means by hard assigning to most probable solution */
    }


    /***** Recalculate (hard-assign) solutions   ******/
    // Aux variables
    ArrayX sumInvRj(nSolFound, zeroX);
    ArrayX sumInvRxj(nSolFound, zeroX);

    sum_p_z.setZero(nSolFound);

    // Hard assign solutions
    for( int j = 0; j < M; j++ )
    {
        p.row(j).maxCoeff(&sNearest[j]);

        sum_p_z(sNearest[j]) += experts[j].get_p_z();
        sumInvRj[sNearest[j]] += fInvRj.col(j);
        sumInvRxj[sNearest[j]] += invRxj.col(j);
    }
    sumW = sum_p_z.array() + pNoiseModelZ;

    // Clear predict data structure
    predictions.clear();

    // Update storage results
    for( int k = 0; k < nSolFound; k++ )
        predictions.push_back( sumInvRxj[k].cwiseQuotient(sumInvRj[k]) );
}

IMLE_CLASS_TEMPLATE
void IMLE_base::getForwardSolutions()
{
    // Aux variables
    ArrayX sumInvRj(nSolFound, zeroX);
    ArrayX sumInvRxj(nSolFound, zeroX);

    // Clear predict data structure
    predictions.clear();
    predictionsVar.clear();
    predictionsWeight.clear();

    for(int j = 0; j < M; j++)
    {
        /* These 3 variables have already been resized in predict() */
        varPhiAuxj(j) = experts[j].getGamma() * experts[j].get_p_z() + sumW[sNearest[j]];
        fInvRj.col(j) = experts[j].getPredXInvVar() * (experts[j].get_p_z() / varPhiAuxj(j));
        invRxj.col(j) = fInvRj.col(j).asDiagonal() * experts[j].getPredX();

        sumInvRj[sNearest[j]] += fInvRj.col(j);
        sumInvRxj[sNearest[j]] += invRxj.col(j);
    }

    // Update storage results
    for( int k = 0; k < nSolFound; k++ )
    {
        if(sum_p_z(k) == 0.0)  // Not likely, but...
        {
            predictions.push_back( zeroX );
            predictionsVar.push_back( infinityX );
            predictionsWeight.push_back( 0.0 );
        }
        else
        {
            predictions.push_back( sumInvRxj[k].cwiseQuotient(sumInvRj[k]) );
            predictionsVar.push_back( sumInvRj[k].cwiseInverse() );
            predictionsWeight.push_back( sum_p_z(k) / sumAll );
        }
    }
}

IMLE_CLASS_TEMPLATE
void IMLE_base::predictInverse(X const &x)
{
    predictInverseSingle(x);

    if( sum_p_x(0) == 0.0 || param.multiValuedSignificance == 0.0)
        return;

    int newSol1, newSol2, worseSol;
    while( !validInverseSolutions(newSol1, newSol2, worseSol) )
    {
        nInvSolFound++;
        clusterInverseSolutions(newSol1, newSol2, worseSol);
        getInverseSolutions();
    }
}

IMLE_CLASS_TEMPLATE
void IMLE_base::predictInverseSingle(X const &x)
{
    // Aux variables
    ZZ sumInvRj = zeroZZ;
    sumInvRzj.setZero(d,1);

    // Clear predict data structure
    invPredictions.clear();
    invPredictionsVar.clear();
    invPredictionsWeight.clear();

    sum_p_x.resize(1);
    sum_p_x(0) = 0.0;
    iInvRj.clear(); iInvRj.reserve(M);
    invRzj.resize(d,M);
    zInvRzj.resize(M);

#ifdef IMLE_NO_TEMPLATES
    // Check dimensions
    if( x.size() != D )
    {
        message("IMLE::predict: query dimension does not match output dimension!!");

        sum_p_x.setZero(1);
    }
    else
#endif
        for(int j = 0; j < M; j++)
        {
            Scal p_x = experts[j].queryX(x);

            // These are needed for multivalued inverse prediction
            iInvRj.push_back(experts[j].getPredZInvVar() * p_x );
            invRzj.col(j) = iInvRj[j] * experts[j].getPredZ();
            zInvRzj(j) = experts[j].getPredZ().dot( invRzj.col(j) );

            sum_p_x(0) += p_x;
            sumInvRj += iInvRj[j];
            sumInvRzj.col(0) += invRzj.col(j);
        }
    sumAll = sum_p_x(0) + pNoiseModelX;

    if(sum_p_x(0) == 0.0)  // This happens when a prediction is sought too far from the current mixture
    {
        invPredictions.push_back(zeroZ);
        invPredictionsVar.push_back( infinityZZ );
        invPredictionsWeight.push_back(0.0);

        nInvSolFound = 1;
        return;
    }
    else if( param.multiValuedSignificance == 0.0 )
    {
        invPredictions.reserve(M);
        invPredictionsVar.reserve(M);
        invPredictionsWeight.reserve(M);

        for(int j = 0; j < M; j++)
        {
            invPredictions.push_back( experts[j].getPredZ() );
            invPredictionsVar.push_back( experts[j].getPredZVar() *(experts[j].get_p_x() + pNoiseModelX) / experts[j].get_p_x() );
            invPredictionsWeight.push_back(experts[j].get_p_x() / sumAll);
        }

        nInvSolFound = M;
        return;
    }
    else
    {
        ZZ invVar = sumInvRj.inverse();

        invPredictions.push_back( invVar * sumInvRzj.col(0) );
        invPredictionsVar.push_back( invVar * sumAll );
        invPredictionsWeight.push_back(sum_p_x(0) / sumAll );
    }

    nInvSolFound = 1;
    sNearestInv.clear();
    sNearestInv.resize(M,0);
}

IMLE_CLASS_TEMPLATE
bool IMLE_base::validInverseSolutions(int &newSol1, int &newSol2, int &worseSol)
{
    Vec T = Vec::Zero(nInvSolFound);
    Vec sumWsq = Vec::Zero(nInvSolFound);
    for(int j = 0; j < M; j++)
    {
        T(sNearestInv[j]) += zInvRzj(j);
        sumWsq(sNearestInv[j]) += experts[j].get_p_x()*experts[j].get_p_x();
    }

    worseSol = -1;
    Scal max_p_value = -1.0, p_value;
    for(int k = 0; k < nInvSolFound; k++)
    {
        T(k) -= invPredictions[k].dot( sumInvRzj.col(k) );
        T(k) /= (sumAll);

        Scal dof = (sum_p_x(k) * sum_p_x(k) / sumWsq(k) - 1.0) * d;
//        Scal dof = (sum_p_x(k) * sum_p_x(k) / sumWsq(k)) * d;

        // Small hack to prevent numerical problems
        dof += 1.0;
        if( !(dof >= 1.0) || T(k) <= 0.0)  // The negation is because of NaN problems
            continue;
//            dof = 1.0;

        if( ( p_value = cdf(boost::math::chi_squared(dof), T(k)) ) >= max_p_value )
        {
            max_p_value = p_value;
            worseSol = k;
        }
    }

    if( max_p_value <= param.multiValuedSignificance )
        return true;
    else
    {
        // Find the largest expert deviation in worse solution
        Scal dist, dMax = -1.0;
        Z aux;
        for(int j = 0; j < M; j++)
        {
            if (sNearestInv[j] != worseSol)
                continue;

            aux = iInvRj[j] * invPredictions[worseSol] - 2.0 * invRzj.col(j);
            dist = zInvRzj(j) + invPredictions[worseSol].dot( aux );

            if( dist > dMax )
            {
                dMax = dist;
                newSol1 = j;
            }
        }

        // Find the largest expert deviation to newSol1
        dMax = -1.0;
        for(int j = 0; j < M; j++)
        {
            if (sNearestInv[j] != worseSol)
                continue;

            aux = iInvRj[j] * experts[newSol1].getPredZ() - 2.0 * invRzj.col(j);
            dist = zInvRzj(j) + experts[newSol1].getPredZ().dot( aux );

            if( dist > dMax )
            {
                dMax = dist;
                newSol2 = j;
            }
        }

        return false;
    }
}

IMLE_CLASS_TEMPLATE
void IMLE_base::clusterInverseSolutions(int newSol1, int newSol2, int worseSol)
{
    Mat p(M,nInvSolFound);
    Mat sol(d,nInvSolFound);
    ArrayZZ sum_invRp(nInvSolFound, zeroZZ);

    // "Educated" guesses for the nInvSolFound solutions
    for(int k = 0; k < nInvSolFound-1; k++)
        sol.col(k) = invPredictions[k];
    // Split worse solution in two
    sol.col(worseSol)    = experts[newSol1].getPredZ();
    sol.col(nInvSolFound-1) = experts[newSol2].getPredZ();

    for( int nIter = 0; nIter < param.iterMax; nIter++ )
    {
        // E-Step
        for( int j = 0; j < M; j++ )
        {
            for( int k = 0; k < nInvSolFound; k++ )
            {
                Z dist = experts[j].getPredZ() - sol.col(k);
                p(j,k) = exp(-0.5*dist.dot(iInvRj[j]*dist)/sumAll);
            }
            Scal pSum;
            if( (pSum = p.row(j).sum()) > 0.0 )
                p.row(j) /= pSum;

            for( int k = 0; k < nInvSolFound; k++ )
                sum_invRp[k] += iInvRj[j] * p(j,k);
        }
        // M-Step
        for( int k = 0; k < nInvSolFound; k++ )
        {
            /* Here I can easily implement k-means by hard assigning to most probable solution */
            sol.col(k).noalias() = sum_invRp[k].inverse() * (invRzj * p.col(k));

            // Cleaning aux variable
            sum_invRp[k].setZero();
        }
    }

    // Hard assign solutions
    for( int j = 0; j < M; j++ )
        p.row(j).maxCoeff(&sNearestInv[j]);
}

IMLE_CLASS_TEMPLATE
void IMLE_base::getInverseSolutions()
{
    // Aux variables
    ArrayZZ sumInvRj(nInvSolFound, zeroZZ);
    sumInvRzj.setZero(d,nInvSolFound);

    // Clear predict data structure
    invPredictions.clear();
    invPredictionsVar.clear();
    invPredictionsWeight.clear();

    sum_p_x.setZero(nInvSolFound);
    for(int j = 0; j < M; j++)
    {
        sum_p_x(sNearestInv[j]) += experts[j].get_p_x();
        sumInvRj[sNearestInv[j]] += iInvRj[j];
        sumInvRzj.col(sNearestInv[j]) += invRzj.col(j);
    }

    // Update storage results
    for( int k = 0; k < nInvSolFound; k++ )
    {
        if(sum_p_x(k) == 0.0)  // Not likely, but...
        {
            invPredictions.push_back(zeroZ);
            invPredictionsVar.push_back( infinityZZ );
            invPredictionsWeight.push_back(0.0);
        }
        else
        {
            ZZ invVar = sumInvRj[k].inverse();

            invPredictions.push_back( invVar * sumInvRzj.col(k) );
            invPredictionsVar.push_back( invVar * (sum_p_x(k) + pNoiseModelX ) );
            invPredictionsWeight.push_back( sum_p_x(k) / sumAll );
        }
    }
}


/*
 * Get's
 */

IMLE_CLASS_TEMPLATE
IMLE_TYPENAME IMLE_base::ArrayXZ const &IMLE_base::getMultiplePredictionsJacobian()
{
    if( !hasPredJacobian )
    {
        predictionsJacobian.clear();
        predictionsVarJacobian.clear();
        predictionsJacobian.resize(nSolFound, zeroXZ);
        predictionsVarJacobian.resize(nSolFound, zeroXZ);
        zeta.resize(d,M);

        Mat weightedKappa = Mat::Zero(d,nSolFound);
        for( int j = 0; j < M; j++ )
            weightedKappa.col(sNearest[j]) += experts[j].get_p_z() * experts[j].getKappa();

        for( int j = 0; j < M; j++ )
        {
            zeta.col(j) = (experts[j].get_p_z() * experts[j].get_dGamma() + sumW(sNearest[j]) * experts[j].getKappa() - weightedKappa.col(sNearest[j])) / varPhiAuxj(j);

            predictionsJacobian[sNearest[j]].noalias() += fInvRj.col(j).asDiagonal() * experts[j].Lambda - invRxj.col(j) * zeta.col(j).transpose();
            predictionsVarJacobian[sNearest[j]].noalias() += fInvRj.col(j) * zeta.col(j).transpose();
        }

        for( int k = 0; k < nSolFound; k++ )
            if(sum_p_z(k) == 0.0)  // Not likely, but...
            {
                predictionsJacobian[k] = zeroXZ;
                predictionsVarJacobian[k] = infinityXZ;
            }
            else
            {
                predictionsJacobian[k].noalias() += predictions[k].asDiagonal() * predictionsVarJacobian[k];
                predictionsJacobian[k] = predictionsVar[k].asDiagonal() * predictionsJacobian[k];
                predictionsVarJacobian[k] = predictionsVar[k].cwiseAbs2().asDiagonal() * predictionsVarJacobian[k];
            }

        hasPredJacobian = true;
    }

    return predictionsJacobian;
}

IMLE_CLASS_TEMPLATE
IMLE_TYPENAME IMLE_base::ArrayXZ const &IMLE_base::getMultiplePredictionErrorReductionDerivative()
{
    if( !hasPredErrorReductionDrvt )
    {
        if( !hasPredJacobian )
            getMultiplePredictionsJacobian();

        predictionsErrorReduction.clear();
        predictionsErrorReductionJacobian.clear();
        predictionsErrorReduction.resize(nSolFound, zeroX);
        predictionsErrorReductionJacobian.resize(nSolFound, zeroXZ);

        ArrayXZ sumInvRjSqPsijDelta_dGammaj(nSolFound, zeroXZ);
        ArrayX sumInvRjSqPsijGammaj(nSolFound, zeroX);
        ArrayXZ sumInvRjSqPsijGammajZetaj(nSolFound, zeroXZ);

        Scal sumH = 0.0;
        for(int j = 0; j < M; j++)
            sumH += experts[j].queryHafterZ( experts[j].getPredX() );

        X invRjSqPsij;
        for(int j = 0; j < M; j++)
        {
            experts[j].queryNewPredXVar( zQuery, experts[j].get_h() / sumH );

            invRjSqPsij = fInvRj.col(j) * (experts[j].get_p_z() / varPhiAuxj(j));
            sumInvRjSqPsijDelta_dGammaj[sNearest[j]].noalias() += invRjSqPsij * ( experts[j].get_dGamma() - experts[j].get_dNewGamma() ).transpose();

            invRjSqPsij *= ( experts[j].getGamma() - experts[j].getNewGamma() ); //Using the same variable just to save resources...
            sumInvRjSqPsijGammaj[sNearest[j]] += invRjSqPsij;
            sumInvRjSqPsijGammajZetaj[sNearest[j]].noalias() += invRjSqPsij * zeta.col(j).transpose();
        }

       for( int k = 0; k < nSolFound; k++ )
            if(sum_p_z(k) == 0.0)  // Not likely, but...
            {
                predictionsErrorReduction[k] = zeroX;
                predictionsErrorReductionJacobian[k] = zeroXZ;
            }
            else
            {
                predictionsErrorReduction[k].noalias() = (predictionsVar[k].cwiseAbs2()).cwiseProduct(sumInvRjSqPsijGammaj[k]);
                predictionsErrorReductionJacobian[k].noalias() = (2.0*predictionsVar[k].cwiseProduct(sumInvRjSqPsijGammaj[k])).asDiagonal() * predictionsVarJacobian[k];
                predictionsErrorReductionJacobian[k].noalias() += predictionsVar[k].cwiseAbs2().asDiagonal() * (sumInvRjSqPsijDelta_dGammaj[k] - 2.0*sumInvRjSqPsijGammajZetaj[k]);
            }

        hasPredErrorReductionDrvt = true;
    }

    return predictionsErrorReductionJacobian;
}



#ifdef __IMLE_TESTER
IMLE_CLASS_TEMPLATE
std::string IMLE_base::getName()
{
    return "IMLE";
}

//template< int d, int D>
//std::string IMLE<d,D, class LinearExpert<d,D> >::getName()
//{
//    return "IMLE";
//}
//
//template< int d, int D>
//std::string IMLE<d,D,FastLinearExpert<d,D> >::getName()
//{
//    return "FastIMLE";
//}

IMLE_CLASS_TEMPLATE
std::string IMLE_base::getInternalState()
{
    std::stringstream state;

    state << "#Models = " << M << ",  Sigma = [" << Sigma.transpose() << "], Psi = [" << Psi.transpose() << "]";

    return state.str();
}

IMLE_CLASS_TEMPLATE
typename IOnlineMixtureOfLinearModels<d,D>::LinearModels IMLE_base::getLinearModels()
{
    int M = getNumberOfModels();
    typename IOnlineMixtureOfLinearModels<d,D>::LinearModels models( M );
    LinearModel<d,D> *modelP;

    for(int i = 0; i<M; i++)
    {
        modelP = &experts[i];
        models[i] = *modelP;
    }

    return models;
}
#endif


/*
 * Print's
 */
IMLE_CLASS_TEMPLATE
void IMLE_base::modelDisplay(std::ostream &out) const
{
   for(int i = 0; i < M; i++)
    {
        out << "------ #" << i+1 << ":" << std::endl;
        experts[i].modelDisplay(out);
        out << "------------------------------------------" << std::endl;
    }

    out << "-----------------------    Common    ----" << std::endl;
    out << EXPAND( M ) << std::endl;
    out << EXPAND(Sigma) << std::endl;
    out << EXPAND(Psi) << std::endl;

}

IMLE_CLASS_TEMPLATE
void IMLE_base::Param::display(std::ostream &out) const
{
    out << EXPAND(d) << std::endl;
    out << EXPAND(D) << std::endl;

    out << EXPAND(alpha) << std::endl;

    out << EXPAND(Psi0) << std::endl;
    out << EXPAND(sigma0) << std::endl;

    out << EXPAND(wsigma) << std::endl;
    out << EXPAND(wSigma) << std::endl;
    out << EXPAND(wNu) << std::endl;
    out << EXPAND(wLambda) << std::endl;
    out << EXPAND(wpsi) << std::endl;
    out << EXPAND(wPsi) << std::endl;

    out << EXPAND(p0) << std::endl;
    out << EXPAND(nOutliers) << std::endl;

    out << EXPAND(multiValuedSignificance) << std::endl;
    out << EXPAND(nSolMin) << std::endl;
    out << EXPAND(nSolMax) << std::endl;
    out << EXPAND(iterMax) << std::endl;

    out << EXPAND(defaultSave) << std::endl;
    out << EXPAND(saveOnExit) << std::endl;
    out << EXPAND(accelerated) << std::endl;

    out << std::endl << std::endl;
}

IMLE_CLASS_TEMPLATE
std::ostream &operator<<(std::ostream &out, IMLE_base const &imle_obj)
{
    imle_obj.displayParameters(out);
    imle_obj.modelDisplay(out);

    return out;
}

