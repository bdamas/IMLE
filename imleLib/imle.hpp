#ifndef __IMLE_H
#define __IMLE_H

#include <boost/serialization/version.hpp>

#include <string>
#include <iostream>
#include <limits>

#define INIT_SIZE  2048
#define MAX_NUMBER_OF_SOLUTIONS 16
#define DEFAULT_SAVE "default.imle"

#ifdef __IMLE_TESTER
    #undef IMLE_NO_TEMPLATES
#endif

#ifdef IMLE_NO_TEMPLATES
    #define IMLE_CLASS_TEMPLATE_HEADER
    #define IMLE_CLASS_TEMPLATE
    #define IMLE_base               IMLE
	#define IMLE_TYPENAME
#else
    #define IMLE_CLASS_TEMPLATE_HEADER  template< int d, int D, template<int,int> class _Expert = ::FastLinearExpert  >
    #define IMLE_CLASS_TEMPLATE         template< int d, int D, template<int,int> class _Expert>
    #define IMLE_base					IMLE<d,D,_Expert>
	#define IMLE_TYPENAME				typename
#endif

#include "EigenSerialized.hpp"
#include "expert.hpp"


IMLE_CLASS_TEMPLATE_HEADER
#ifdef __IMLE_TESTER
class IMLE : public IOnlineMultivaluedMixtureOfLinearModels<d,D>
#else
class IMLE
#endif
{
public:
#ifdef IMLE_NO_TEMPLATES
    const int d;
    const int D;

    typedef Vec Z;
    typedef Vec X;
    typedef Mat ZZ;
    typedef Mat XZ;
    typedef Mat XX;

    typedef ArrayVec ArrayZ;
    typedef ArrayVec ArrayX;
    typedef ArrayMat ArrayZZ;
    typedef ArrayMat ArrayXX;
    typedef ArrayMat ArrayXZ;
#else
    typedef typename Eig<d,D>::Z Z;
    typedef typename Eig<d,D>::X X;
    typedef typename Eig<d,D>::ZZ ZZ;
    typedef typename Eig<d,D>::XZ XZ;
    typedef typename Eig<d,D>::XX XX;

    typedef typename Eig<d,D>::ArrayZ ArrayZ;
    typedef typename Eig<d,D>::ArrayX ArrayX;
    typedef typename Eig<d,D>::ArrayZZ ArrayZZ;
    typedef typename Eig<d,D>::ArrayXX ArrayXX;
    typedef typename Eig<d,D>::ArrayXZ ArrayXZ;
#endif

    struct Param {
    #ifdef IMLE_NO_TEMPLATES
        int d;
        int D;
        Param(int _d = 1, int _D = 1);
    #else
        Param();
    #endif
        // Parameters
        bool accelerated;
        Scal alpha;

        X Psi0;
        Scal sigma0;
        Scal wsigma;
        Scal wpsi;

        Scal wSigma;
        Scal wNu;
        Scal wLambda;
        Scal wPsi;

        int nOutliers;
        Scal p0;

        Scal multiValuedSignificance;
    //	bool predictWithOutlierModel;
        int nSolMin;
        int nSolMax;
        int iterMax;

        // Saving
        bool saveOnExit;
        std::string defaultSave;

        void display(std::ostream &out = std::cout) const;

        // Boost serialization
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version);
    };

#ifdef IMLE_NO_TEMPLATES
    typedef ::FastLinearExpert Expert;
#else
    typedef _Expert<d,D> Expert;
#endif
    typedef std::vector< Expert, Eigen::aligned_allocator<Expert> > Experts;

	// Constructors and destructor
#ifdef IMLE_NO_TEMPLATES
    IMLE(int d, int D, int pre_alloc = INIT_SIZE);
    IMLE(int d, int D, Param const &prm, int pre_alloc = INIT_SIZE);
	IMLE(int d, int D, std::string const &filename, int pre_alloc = INIT_SIZE);
#else
    IMLE(Param const &prm = Param(), int pre_alloc = INIT_SIZE);
	IMLE(std::string const &filename, int pre_alloc = INIT_SIZE);
#endif
	~IMLE();
	void reset();
	void reset(Param const &prm);

    // Algorithm
	void update(Z const &z, X const &x);

	X const &predict(Z const &z);               	// Single Forward Prediction
	X const &predictStrongest(Z const &z);          // Strongest Forward Prediction
	void predictMultiple(Z const &z);  	            // Multi Forward Prediction

//	void predictInverseSingle(X const &x);	    // Single Inverse Prediction
//	void predictInverseStrongest(X const &x);	// Strongest Inverse Prediction
	void predictInverse(X const &x);	            // Multiple Inverse Prediction
//	void predictInverseImproved(X const &x);	    // Multiple Inverse Prediction

	X const &getPrediction() {
        return predictions[solIdx]; }
	X const &getPredictionVar() {
        return predictionsVar[solIdx]; }
	Scal getPredictionWeight() {
        return predictionsWeight[solIdx]; }
	XZ const &getPredictionJacobian() {
	    getMultiplePredictionsJacobian();
        return predictionsJacobian[solIdx]; }
	X const &getPredictionErrorReduction() {
        getMultiplePredictionErrorReductionDerivative();
        return predictionsErrorReduction[solIdx]; }
	XZ const &getPredictionErrorReductionDerivative() {
        getMultiplePredictionErrorReductionDerivative();
        return predictionsErrorReductionJacobian[solIdx]; }

	ArrayX const &getMultiplePredictions() {
        return predictions; }
	ArrayX const &getMultiplePredictionsVar() {
        return predictionsVar; }
	ArrayScal const &getMultiplePredictionsWeight() {
        return predictionsWeight; }
    int getNumberOfSolutionsFound() {return nSolFound; }
	ArrayXZ const &getMultiplePredictionsJacobian();
	ArrayX const &getMultiplePredictionErrorReduction() {
	    getMultiplePredictionErrorReductionDerivative();
        return predictionsErrorReduction; }
	ArrayXZ const &getMultiplePredictionErrorReductionDerivative();

	ArrayZ const &getInversePredictions() {
        return invPredictions; }
	ArrayZZ const &getInversePredictionsVar() {
        return invPredictionsVar; }
	ArrayScal const &getInversePredictionsWeight() {
        return invPredictionsWeight;}
    int getNumberOfInverseSolutionsFound() {
        return nInvSolFound; }

    // Parameter display
	int inputDim() {
		return d; }
	int outputDim() {
		return D; }
#ifdef IMLE_NO_TEMPLATES
    static void createDefaultParamFile(int _d, int _D, std::string const &fname);
#else
    static void createDefaultParamFile(std::string const &fname);
#endif
	void setParameters(Param const &prm);
	Param const &getParameters() {
        return param; }
	bool loadParameters(std::string const &fname);
	void displayParameters(std::ostream &out = std::cout) const {
        param.display(out); }

    // Save and load
	bool save(std::string const &filename);
	bool load(std::string const &filename);

    // Get internal model
    int getNumberOfModels() {
        return M; }
	Experts const &getExperts() {
        return experts; }
	Z const &getSigma() {
        return Sigma; }
	X const &getPsi() {
        return Psi; }
	void modelDisplay(std::ostream &out = std::cout) const;

	// Used by evaluateLearner
#ifdef __IMLE_TESTER
    // Identification & State
    inline std::string getName();
	inline std::string getInternalState();
    typename IOnlineMixtureOfLinearModels<d,D>::LinearModels getLinearModels();
#endif

protected:
    // Save, load, init, params, display...
	void message(std::string const &msg) {
        std::cout << msg << std::endl; }

	void init( int pre_alloc );
	void set_internal_parameters();
    //      Boost serialization
    friend class boost::serialization::access;
    template<class Archive>
    inline void serialize(Archive & ar, const unsigned int version);

    // Algorithm
    bool createNewExpert(Z const &z, X const &x);
	void EM(Z const &z, X const &x);

	bool validForwardSolutions(int &newSol1, int &newSol2, int &worseSol);
	void clusterForwardSolutions(int newSol1, int newSol2, int worseSol);
	void getForwardSolutions();

    void predictInverse2(X const &x);
	void predictInverseSingle(X const &x);
	bool validInverseSolutions(int &newSol1, int &newSol2, int &worseSol);
	void clusterInverseSolutions(int newSol1, int newSol2, int worseSol);
	void getInverseSolutions();

	/*
	* CLASS DATA
	*/

	// Parameters
	Param param;

    // Experts
    Experts experts;
    int M;

    // Common priors
    Z Sigma;
    X Psi;

    // Internal Parameters
	int noise_to_go;
	Scal sig_level_multi_test;
    Scal pNoiseModelZ, pNoiseModelX, pNoiseModelZX;
    Scal sig_level_noiseZ_rbf, sig_level_noiseX_rbf, sig_level_noiseZX_rbf;

	// Storage results
	Z zQuery;
	std::vector<int> sNearest;
	Vec varPhiAuxj;
	Mat fInvRj;
	Mat invRxj;
	Vec sum_p_z, sumW;
    Scal sumAll;
    Mat zeta;
	ArrayX predictions;
	ArrayX predictionsVar;
	ArrayScal predictionsWeight;
	ArrayXZ predictionsJacobian;
	ArrayXZ predictionsVarJacobian;
	ArrayX predictionsErrorReduction;
	ArrayXZ predictionsErrorReductionJacobian;

	int solIdx;
	int nSolFound;
	bool hasPredJacobian;
	bool hasPredErrorReductionDrvt;

	std::vector<int> sNearestInv;
	ArrayZZ iInvRj;
	Mat invRzj;
	Vec zInvRzj;
	Vec sum_p_x;
	Mat sumInvRzj;
	ArrayZ invPredictions;
	ArrayZZ invPredictionsVar;
	ArrayScal invPredictionsWeight;
	int nInvSolFound;

	// To speed things up
	ZZ zeroZZ;
	ZZ infinityZZ;
	Z zeroZ;
	X zeroX;
	X infinityX;
	XZ zeroXZ;
	XZ infinityXZ;
};

//IMLE_CLASS_TEMPLATE
//std::ostream &operator<<(std::ostream &out, typename IMLE<d,D,_Expert>::Param const &param);

IMLE_CLASS_TEMPLATE
std::ostream &operator<<(std::ostream &out, IMLE_base const &imle_obj);


// Boost serialization
IMLE_CLASS_TEMPLATE
template<class Archive>
void IMLE_base::Param::serialize(Archive & ar, const unsigned int version)
{
    // Dimensionality check
    int dd = d, DD = D;
    ar & BOOST_SERIALIZATION_NVP(dd);
    ar & BOOST_SERIALIZATION_NVP(DD);
    if( dd != d || DD != D )
    {
        std::cerr << "IMLE: Dimensions do not agree when loading file!\n";
        return;
    }

    ar & BOOST_SERIALIZATION_NVP(alpha);

    ar & BOOST_SERIALIZATION_NVP(Psi0);
    ar & BOOST_SERIALIZATION_NVP(sigma0);

    ar & BOOST_SERIALIZATION_NVP(wsigma);
    ar & BOOST_SERIALIZATION_NVP(wSigma);
    ar & BOOST_SERIALIZATION_NVP(wNu);
    ar & BOOST_SERIALIZATION_NVP(wLambda);
    ar & BOOST_SERIALIZATION_NVP(wpsi);
    ar & BOOST_SERIALIZATION_NVP(wPsi);

    ar & BOOST_SERIALIZATION_NVP(nOutliers);
    ar & BOOST_SERIALIZATION_NVP(p0);

    ar & BOOST_SERIALIZATION_NVP(multiValuedSignificance);
    ar & BOOST_SERIALIZATION_NVP(nSolMin);
    ar & BOOST_SERIALIZATION_NVP(nSolMax);
    ar & BOOST_SERIALIZATION_NVP(iterMax);

    ar & BOOST_SERIALIZATION_NVP(saveOnExit);
    ar & BOOST_SERIALIZATION_NVP(defaultSave);
    ar & BOOST_SERIALIZATION_NVP(accelerated);
}


/*
 * typedefs
 */

// Only allowed in C++0x
//template< int d, int D>
//using IMLE = IMLE< d, D, LinearExpert<d,D> >;
//template< int d, int D>
//using FastIMLE = IMLE< d, D, FastLinearExpert<d,D> >;


// IMLE template implementation
#ifndef IMLE_NO_TEMPLATES
    #include "imleInline.hpp"
#endif



#endif

