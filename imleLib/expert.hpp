#ifndef __EXPERT_H
#define __EXPERT_H

#include "EigenSerialized.hpp"
#include "imle.hpp"

#ifdef __IMLE_TESTER
    #undef IMLE_NO_TEMPLATES
#endif

#ifdef IMLE_NO_TEMPLATES
    #define EXPERT_CLASS_TEMPLATE
    #define LINEAR_EXPERT_base               LinearExpert
    #define FAST_LINEAR_EXPERT_base          FastLinearExpert
#else
    #define EXPERT_CLASS_TEMPLATE            template< int d, int D>
    #define LINEAR_EXPERT_base               LinearExpert<d,D>
    #define FAST_LINEAR_EXPERT_base          FastLinearExpert<d,D>
#endif

IMLE_CLASS_TEMPLATE
class IMLE;

/*
 * LinearExpert Interface
 */

EXPERT_CLASS_TEMPLATE
#ifdef __IMLE_TESTER
class LinearExpert : public LinearModel<d,D>
#else
class LinearExpert
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
#else
    typedef typename Eig<d,D>::Z Z;
    typedef typename Eig<d,D>::X X;
    typedef typename Eig<d,D>::ZZ ZZ;
    typedef typename Eig<d,D>::XZ XZ;
    typedef typename Eig<d,D>::XX XX;
#endif

    // Constructors
#ifdef IMLE_NO_TEMPLATES
    LinearExpert(int d, int D, Z const &z, X const &x, IMLE *mixture);
    LinearExpert(int d, int D);                                                 // Needed for Boost Serialization...
    LinearExpert & operator=(LinearExpert const &other);  // This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
#else
    LinearExpert(Z const &z, X const &x, IMLE<d,D,::LinearExpert> *mixture);
    LinearExpert();                                                 // Needed for Boost Serialization...
    LinearExpert<d,D> & operator=(LinearExpert<d,D> const &other);  // This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
#endif
    // Update
    void e_step( Z const &z, X const &x, Scal h );
    void m_step();

    // Predict
    Scal queryH( Z const &z, X const &x );
    Scal queryZ( Z const &z);
    Scal queryXafterZ( X const &x );
    Scal queryZX( Z const &z, X const &x );
    Scal queryHafterZ( X const &x );
    Scal queryHafterZX();
    Scal queryZXandH( Z const &z, X const &x );
    Scal queryX( X const &x);

    // Gets and Sets
    inline Scal get_h() const;
    inline Scal get_p_z() const;
    inline Scal get_p_x() const;
    inline Scal get_p_zx() const;
    inline Scal get_rbf_zx() const;

    inline X const &getPredX() const;
    inline X const &getPredXVar() const;
    inline X const &getPredXInvVar() const;
    inline Scal getGamma() const;
    inline Z const &get_dGamma() const;

    inline Z const &getKappa() const;
    inline Scal getNewGamma() const;
    inline Z const &get_dNewGamma() const;

    inline Z const &getPredZ() const;
    inline ZZ const &getPredZVar() const;
    inline ZZ const &getPredZInvVar() const;
    inline ZZ const &getUncertFactorPredZ() const;

    // Display
	void modelDisplay(std::ostream &out = std::cout) const;

#ifdef __IMLE_TESTER
    using LinearModel<d,D>::Nu;
    using LinearModel<d,D>::Mu;
    using LinearModel<d,D>::Lambda;
    using LinearModel<d,D>::invSigma;
    using LinearModel<d,D>::Psi;
#else
    Z Nu;
    X Mu;
    XZ Lambda;
    ZZ invSigma;
    X Psi;
#endif
    X invPsi;

protected:
    // Boost serialization
    friend class boost::serialization::access;
    template<typename Archive>
    inline void serialize(Archive & ar, const unsigned int version);

    // Shared prior
//    IMLE<d,D,::LinearExpert> *mixture;
#ifdef IMLE_NO_TEMPLATES
	IMLE *mixture;
#else
    IMLE<d,D,::LinearExpert> *mixture;
#endif
    // Memory Traces
    Scal H;        // Needed to decay statistics
    Scal Sh;
    Z Sz;
    X Sx;
    XZ Sxz;
    ZZ Szz;
    X Sxx;

    // Priors parameters and decay
    Z Nu0;
    Scal alpha;
    Scal wPsi;
    Scal wNu;
    Scal wSigma;
    Scal wLambda;

    // Recomputing
    bool recompute;

    // Storing results
    Scal h, p_z_T, p_x_Norm, p_zx, rbf_z, rbf_x, rbf_zx;

    X pred_x;
    Z pred_z;

    Scal gamma;
    Z dGamma;
    Scal newGamma;
    Z dNewGamma;
    Z kappa;

    ZZ pred_z_var_factor;
    ZZ pred_z_var;
    ZZ pred_z_invVar;

	// Aux variables
    Z dz;
    X dx;
    Scal dz_invSigma_dz;

    Scal sqrtDetInvSigma, sqrtDetInvPsi;
    ZZ varLambda;

    ZZ Sigma;
    XZ PsiLambda;
    ZZ LambdaPsiLambda;
    XX p_x_invVar;
    Scal p_x_invVarSqrtDet;
};

/*
 * FastLinearExpert Interface
 */
EXPERT_CLASS_TEMPLATE
class FastLinearExpert : public LINEAR_EXPERT_base
{
public:
    // Inherited from base class LinearExpert<d,D>
    typedef IMLE_TYPENAME LINEAR_EXPERT_base::Z Z;
    typedef IMLE_TYPENAME LINEAR_EXPERT_base::X X;
    typedef IMLE_TYPENAME LINEAR_EXPERT_base::ZZ ZZ;
    typedef IMLE_TYPENAME LINEAR_EXPERT_base::XZ XZ;
    typedef IMLE_TYPENAME LINEAR_EXPERT_base::XX XX;

#ifdef IMLE_NO_TEMPLATES
    using LINEAR_EXPERT_base::d;
    using LINEAR_EXPERT_base::D;

	FastLinearExpert(int d, int D, Z const &z, X const &x, IMLE *_mixture);
    FastLinearExpert(int d, int D);
    FastLinearExpert & operator=(FastLinearExpert const &other);  // This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
#else
	FastLinearExpert(Z const &z, X const &x, IMLE<d,D,::FastLinearExpert> *_mixture);
    FastLinearExpert();
    FastLinearExpert<d,D> & operator=(FastLinearExpert<d,D> const &other);  // This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
#endif

    void e_step( Z const &z, X const &x, Scal h );
    void m_step();

    void queryNewPredXVar( Z const &z, Scal hNorm );

    using LINEAR_EXPERT_base::Nu;
    using LINEAR_EXPERT_base::Mu;
    using LINEAR_EXPERT_base::Lambda;
    using LINEAR_EXPERT_base::invSigma;
    using LINEAR_EXPERT_base::Psi;
    using LINEAR_EXPERT_base::invPsi;

    void modelDisplay(std::ostream &out = std::cout) const;

protected:
    // Boost serialization
    friend class boost::serialization::access;
    template<typename Archive>
    inline void serialize(Archive & ar, const unsigned int version);

    // Shared prior
#ifdef IMLE_NO_TEMPLATES
	IMLE *mixture;
#else
    IMLE<d,D,::FastLinearExpert> *mixture;
#endif

    // Memory Traces
    using LINEAR_EXPERT_base::H;
    using LINEAR_EXPERT_base::Sh;
    using LINEAR_EXPERT_base::Sz;
    using LINEAR_EXPERT_base::Sx;
    using LINEAR_EXPERT_base::Sxz;
    using LINEAR_EXPERT_base::Sxx;
    ZZ invSzz;
    ZZ invSzz0;
    Scal detInvSzz0;

    // Priors parameters and decay
    using LINEAR_EXPERT_base::Nu0;
    using LINEAR_EXPERT_base::alpha;
    using LINEAR_EXPERT_base::wPsi;
    using LINEAR_EXPERT_base::wNu;
    using LINEAR_EXPERT_base::wSigma;
    using LINEAR_EXPERT_base::wLambda;

    // Recomputing
    using LINEAR_EXPERT_base::recompute;

	// Aux variables
    using LINEAR_EXPERT_base::sqrtDetInvSigma;
    using LINEAR_EXPERT_base::sqrtDetInvPsi;
    using LINEAR_EXPERT_base::varLambda;

    // Active learning
    using LINEAR_EXPERT_base::gamma;
    using LINEAR_EXPERT_base::dGamma;
    using LINEAR_EXPERT_base::newGamma;
    using LINEAR_EXPERT_base::dNewGamma;
};


// Serialization issues with IMLE_NO_TEMPLATES
//
#ifdef IMLE_NO_TEMPLATES
namespace boost { namespace serialization {

template<class Archive>
inline void save_construct_data(
    Archive & ar, const LinearExpert * e, const unsigned int file_version
){
    // save data required to construct instance
    ar << e->d;
    ar << e->D;
}

template<class Archive>
inline void load_construct_data(
    Archive & ar, LinearExpert * e, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
    int d, D;
    ar >> d;
    ar >> D;
    // invoke inplace constructor to initialize instance of my_class
    ::new(e) LinearExpert(d,D);
}

template<class Archive>
inline void save_construct_data(
    Archive & ar, const FastLinearExpert * e, const unsigned int file_version
){
    // save data required to construct instance
    ar << e->d;
    ar << e->D;
}

template<class Archive>
inline void load_construct_data(
    Archive & ar, FastLinearExpert * e, const unsigned int file_version
){
    // retrieve data from archive required to construct new instance
    int d, D;
    ar >> d;
    ar >> D;
    // invoke inplace constructor to initialize instance of my_class
    ::new(e) FastLinearExpert(d,D);
}

}} // namespace ...
#endif



// Expert template implementation
#ifndef IMLE_NO_TEMPLATES
    #include "expertInline.hpp"
#endif

#endif


