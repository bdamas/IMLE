#include <Eigen/LU>
#include <iostream>
#include <boost/math/special_functions/gamma.hpp>

#define EXPAND(var) #var "= " << var
#define EXPAND_N(var) #var "=\n" << var

#define EPSILON 0.000001


/*
 * LinearExpert Constructors
 */

EXPERT_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
LINEAR_EXPERT_base::LinearExpert(int d, int D, Z const &z, X const &x, IMLE *_mixture)
: d(d), D(D)
#else
LINEAR_EXPERT_base::LinearExpert(Z const &z, X const &x, IMLE<d,D,::LinearExpert> *_mixture)
#endif
{
    mixture = _mixture;

    alpha = mixture->getParameters().alpha;
    wPsi = mixture->getParameters().wPsi;
    wNu = mixture->getParameters().wNu;
    wSigma = mixture->getParameters().wSigma;
    wLambda = mixture->getParameters().wLambda;

    Nu0 = z;


    H = EPSILON;

    Sh = 0.0;
    Sz.setZero(d);
    Sx.setZero(D);
    Sxz.setZero(D,d);
    Sxx.setZero(D);
    Szz.setIdentity(d,d) *= EPSILON;


    Nu = z;
    Mu = x;
    Lambda.setZero(D,d);
    varLambda = 1.0/wLambda * ZZ::Identity(d,d);

    Z sigma0 = wSigma/(wSigma+d+2.0)*mixture->getSigma();
    Sigma = sigma0.asDiagonal();
    invSigma = sigma0.cwiseInverse().asDiagonal();
    sqrtDetInvSigma = sqrt( 1.0 / sigma0.prod() );
//    Sigma.setIdentity(d,d) *=  wSigma*mixture->sigma/(wSigma+d+2.0);
//    invSigma.setIdentity(d,d) *=  (wSigma+d+2.0)/(wSigma*mixture->sigma);
//    sqrtDetInvSigma = pow((wSigma+d+2.0)/(wSigma*mixture->sigma), d/2.0);
    Psi = mixture->getPsi() / (1.0 + 2.0/wPsi) ;
    invPsi = Psi.cwiseInverse();
    sqrtDetInvPsi = sqrt(invPsi.prod());


    recompute = true;
}

EXPERT_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
LINEAR_EXPERT_base::LinearExpert(int d, int D)
: d(d), D(D)
#else
LINEAR_EXPERT_base::LinearExpert()
#endif
{
    recompute = true;
}

// This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
EXPERT_CLASS_TEMPLATE
LINEAR_EXPERT_base & LINEAR_EXPERT_base::operator=(LINEAR_EXPERT_base const &other)
{
    std::cerr << "LINEAR_EXPERT_base::operator=(LINEAR_EXPERT_base const &other) --> YOU SHOULDN'T BE SEEING THIS!!\n";
    abort();

    return *this;
}

/*
 * LinearExpert Algorithm
 */
EXPERT_CLASS_TEMPLATE
void LINEAR_EXPERT_base::e_step( Z const &z, X const &x, Scal h )
{
    Scal decay = (pow(H+h,alpha) - h) / pow(H,alpha);

    H += h;

    Z zh = z * h;
	X xh = x * h;

	Sh  *= decay; Sh  += h;
	Sz  *= decay; Sz  += zh;
	Sx  *= decay; Sx  += xh;
	Sxz *= decay; Sxz.noalias() += xh * z.transpose();
	Szz *= decay; Szz.noalias() += zh * z.transpose();
	Sxx *= decay; Sxx += xh.cwiseProduct(x);

    wPsi *= decay;
    wNu *= decay;
    wSigma *= decay;
    wLambda *= decay;
}

EXPERT_CLASS_TEMPLATE
void LINEAR_EXPERT_base::m_step()
{
    Z meanZ = Sz / Sh;
    X meanX = Sx / Sh;

    if( wNu == 0.0 )
    {
        Nu = meanZ;
        Sigma = Szz;
        Sigma += (wSigma*mixture->getSigma()).asDiagonal();
        Sigma.noalias() -= Sz*Nu.transpose();
        Sigma /= (Sh + wSigma + d + 1.0);
    }
    else
    {
        Nu = (wNu * Nu0 + Sz) / (wNu + Sh);
        Sigma = Szz;
        Sigma += (wSigma*mixture->getSigma()).asDiagonal();
        Sigma.noalias() += (wNu*Nu0)*Nu0.transpose();
        Sigma.noalias() -= ((wNu+Sh)*Nu)*Nu.transpose();
        Sigma /= (Sh + wSigma + d + 2.0);
    }
	invSigma = Sigma.inverse();
    sqrtDetInvSigma = sqrt(invSigma.determinant());

    ZZ zz = Szz;
    zz += (wLambda * Z::Ones(d)).asDiagonal();
    zz.noalias() -= meanZ*Sz.transpose();
    varLambda = zz.inverse();
    XZ xz = Sxz;
    xz.noalias() -= meanX*Sz.transpose();
    Lambda.noalias() = xz * varLambda;
    Mu = meanX;
    Mu.noalias() += Lambda * (Nu - meanZ);

    Psi = wPsi * mixture->getPsi();
    Psi.noalias() += Sxx - meanX.cwiseProduct(Sx) - xz.cwiseProduct(Lambda).rowwise().sum();
    Psi /= (wPsi + Sh + 2.0 );
    invPsi = Psi.cwiseInverse();
    sqrtDetInvPsi = sqrt(invPsi.prod());

    recompute = true;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryH( Z const &z, X const &x )
{
    dz = z - Nu;
    pred_x = Lambda * dz + Mu;
    dx = x - pred_x;

    h = exp(-0.5*dz.dot(invSigma * dz)) * sqrtDetInvSigma * exp(-0.5*dx.cwiseAbs2().dot(invPsi)) * sqrtDetInvPsi;

    return h;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryZ( Z const &z )
{
    dz = z - Nu;
    pred_x = Lambda * dz + Mu;

    Scal dof = wSigma + Sh - d + 1.0;
    Scal aux1 = (wNu + Sh + 1.0) / (wNu + Sh) * (wSigma + Sh + d + (wNu == 0.0 ? 1.0 : 2.0));

    Z aux2 = invSigma * dz;
    dz_invSigma_dz = dz.dot( aux2 );
    Scal aux3 = 1.0 + dz_invSigma_dz / aux1;
    kappa = ( (dof+d)/aux1/aux3 ) * aux2;

    rbf_z = pow(aux3 , -0.5*(dof+d) );
    p_z_T = sqrtDetInvSigma / boost::math::tgamma_delta_ratio(0.5*dof, 0.5*d) * pow(2.0/aux1, 0.5*d) * rbf_z;

    Z dzAux = z - Sz/Sh;
    dGamma = varLambda * dzAux;
    gamma = 1.0/Sh + dzAux.dot( dGamma );
    dGamma *= 2.0;

    return p_z_T;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryXafterZ( X const &x )
{
    dx = x - pred_x;

    X xInvVar = invPsi / (1.0 + gamma);

    rbf_x = exp(-0.5*dx.cwiseAbs2().dot(xInvVar));
    rbf_zx = rbf_z * rbf_x;
    p_zx = p_z_T * rbf_x * sqrt(xInvVar.prod());

    return p_zx;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryZX( Z const &z, X const &x )
{
    queryZ(z);
    return queryXafterZ( x );
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryHafterZ( X const &x )
{
    dx = x - pred_x;

    h = exp(-0.5*dz.dot(invSigma * dz)) * sqrtDetInvSigma * exp(-0.5*dx.cwiseAbs2().dot(invPsi)) * sqrtDetInvPsi;

    return h;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryHafterZX()
{
    h = exp(-0.5*dz.dot(invSigma * dz)) * sqrtDetInvSigma * exp(-0.5*dx.cwiseAbs2().dot(invPsi)) * sqrtDetInvPsi;

    return h;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryZXandH( Z const &z, X const &x )
{
    queryZX( z, x );
    queryHafterZX();

    return p_zx;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::queryX( X const &x )
{
    if( recompute )
    {
        PsiLambda = invPsi.asDiagonal() * Lambda;
        LambdaPsiLambda = Lambda.transpose() * PsiLambda;

        // Observation Variance
        pred_z_invVar = invSigma + LambdaPsiLambda;
        pred_z_var = pred_z_invVar.inverse();                      //TODO: Optimize this!

        p_x_invVar = - PsiLambda * pred_z_var * PsiLambda.transpose();
        p_x_invVar += invPsi.asDiagonal();
        p_x_invVarSqrtDet = sqrt(p_x_invVar.determinant());        //TODO: Optimize this!

        // Prediction Variance
//        pred_z_var_factor = pred_z_var * LambdaPsiLambda * pred_z_var;
//        if( wNu == 0.0 )
//            pred_z_var *= 1.0/Sh;
//        else
//        {
//            Z dz = Nu - Sz/Sh;
//            pred_z_var *= 1.0/Sh + dz.dot( varLambda * dz );
//        }

        recompute = false;
    }

    X dx = x - Mu;

    pred_z = Nu + pred_z_var * PsiLambda.transpose() * dx;
    p_x_Norm = p_x_invVarSqrtDet * exp( -0.5 * dx.dot(p_x_invVar * dx) );

    return p_x_Norm;
}


/*
 * LinearExpert Display
 */
EXPERT_CLASS_TEMPLATE
void LINEAR_EXPERT_base::modelDisplay(std::ostream &out) const
{
    out << EXPAND_N(Nu) << std::endl;
    out << EXPAND_N(invSigma) << std::endl;
    out << EXPAND_N(Mu) << std::endl;
    out << EXPAND_N(Lambda) << std::endl;
    out << EXPAND_N(Psi) << std::endl;

//    out << EXPAND_N(H) << std::endl;
//    out << EXPAND_N(Sh) << std::endl;
//    out << EXPAND_N(Sz) << std::endl;
//    out << EXPAND_N(Sx) << std::endl;
//    out << EXPAND_N(Sxz) << std::endl;
//    out << EXPAND_N(Szz) << std::endl;
//    out << EXPAND_N(Sxx) << std::endl;
//    out << EXPAND_N(Nu0) << std::endl;
//
//    out << EXPAND(wPsi) << std::endl;
//    out << EXPAND(wNu) << std::endl;
//    out << EXPAND(wSigma) << std::endl;
//    out << EXPAND(wLambda) << std::endl;
//    out << EXPAND(alpha) << std::endl;
//    out << EXPAND(mixture->getPsi()) << std::endl;
//    out << EXPAND(mixture->getSigma()) << std::endl;
}




EXPERT_CLASS_TEMPLATE
void FAST_LINEAR_EXPERT_base::queryNewPredXVar(Z const &z, Scal hNorm)
{
    Scal decay = (pow(H+hNorm,alpha) - hNorm) / pow(H,alpha);

    Scal newSh = decay * Sh + hNorm;
    Z newSz = decay * Sz + hNorm * z;

    Z f,g;
    f.noalias() = invSzz * z;
    g.noalias() = invSzz * newSz;

    Scal a = f.dot( z );
    Scal b = g.dot( z );
    Scal c = g.dot( newSz );

    Scal dd = decay / hNorm + a;
    Scal e = 1.0 - a / dd;
    Scal i = b / dd;
    Scal k = c - b*i;
    Scal m = b*e*newSh - k;
    Scal n = decay * newSh - k;

    newGamma = 1.0 / newSh + (a*e + ((k+m*m/n)/newSh-2*b*e)/newSh)/decay;

    Scal o = (m/n - 1.0)/newSh;
    Scal p = (hNorm-i)*o*o;

    dNewGamma = (2.0*(e*(e+(hNorm-2*i)*o)-i*p)/decay) * f + (2.0*(e*o+p)/decay)*g;
}


/*
 * FastLinearExpert Constructors
 */
EXPERT_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
FAST_LINEAR_EXPERT_base::FastLinearExpert(int d, int D, Z const &z, X const &x, IMLE *_mixture)
: LINEAR_EXPERT_base(d,D)
#else
FAST_LINEAR_EXPERT_base::FastLinearExpert(Z const &z, X const &x, IMLE<d,D,::FastLinearExpert> *_mixture)
#endif
{
    mixture = _mixture;

    alpha = mixture->getParameters().alpha;
    wPsi = mixture->getParameters().wPsi;
    wNu = mixture->getParameters().wNu;
    wSigma = mixture->getParameters().wSigma + d + ((wNu==0) ? 1.0 : 2.0);
    wLambda = mixture->getParameters().wLambda;

    Nu0 = z;


    H = EPSILON;

    Sh = 0.0;
    Sz.setZero(d);
    Sx.setZero(D);
    Sxz.setZero(D,d);
    Sxx = wPsi * mixture->getPsi();
//Sxx = X::Zero(D);
    invSzz.setIdentity(d,d) /= wLambda;
    Z Sigma0 = mixture->getParameters().wSigma*mixture->getSigma();
    invSzz0 = Sigma0.cwiseInverse().asDiagonal();
	// Using Moore-Penrose Rank-1 update
	Z SZ = invSzz0 * Nu0;
	Scal DOT = Nu0.dot(SZ);
	Scal DEN = DOT + 1 / wNu;
	invSzz0 -= (SZ/DEN) * SZ.transpose();

	// Using Determinant Rank-1 update
	detInvSzz0 = 1.0 / Sigma0.prod();
	detInvSzz0 /= (DOT*wNu + 1.0);

    Nu = z;
    Mu = x;
    Lambda.setZero(D,d);
    invSigma = (Sigma0/wSigma).cwiseInverse().asDiagonal();
    sqrtDetInvSigma = sqrt( 1.0 / (Sigma0/wSigma).prod() );

    Psi = mixture->getPsi() / (1.0 + 2.0/wPsi) ;
    invPsi = Psi.cwiseInverse();
    sqrtDetInvPsi = sqrt(invPsi.prod());

//    invSzz.setIdentity(d,d) /= EPSILON;
//    invSzz0.setIdentity(d,d) /= this->w0 * this->sigma;
//    detInvSzz0 = 1.0 / pow(this->w0 * this->sigma, d);
//
//    this->Nu = z;
//    this->Mu = x;
//    this->Lambda.setZero(D,d);
//    this->invSSEzz.setZero(d,d);
//
//    this->invSigma.setIdentity(d,d) *=  (this->w0+d+1.0)/(this->w0*this->sigma);
//    this->sqrtDetInvSigma = pow((this->w0+d+1.0)/(this->w0*this->sigma), d/2.0);
//
//    this->recompute = true;
    recompute = true;
}

EXPERT_CLASS_TEMPLATE
#ifdef IMLE_NO_TEMPLATES
FAST_LINEAR_EXPERT_base::FastLinearExpert(int d, int D)
: LINEAR_EXPERT_base(d,D)
#else
FAST_LINEAR_EXPERT_base::FastLinearExpert()
: LINEAR_EXPERT_base()
#endif
{
}

// This is due to a STL annoyance (see http://blog.copton.net/archives/2007/10/13/stdvector/index.html)
EXPERT_CLASS_TEMPLATE
FAST_LINEAR_EXPERT_base & FAST_LINEAR_EXPERT_base::operator=(FAST_LINEAR_EXPERT_base const &other)
{
    std::cerr << "FAST_LINEAR_EXPERT_base::operator=(FAST_LINEAR_EXPERT_base const &other) --> YOU SHOULDN'T BE SEEING THIS!!\n";
    abort();

    return *this;
}


 /*
  * FastLinearExpert Algorithm
  */
EXPERT_CLASS_TEMPLATE
void FAST_LINEAR_EXPERT_base::e_step( Z const &z, X const &x, Scal h )
{
	Z SZ;
	Scal DOT, DEN;

    Scal decay = (pow(H+h,alpha) - h) / pow(H,alpha);

    H += h;

    Z zh = z * h;
	X xh = x * h;

	Sh  *= decay; Sh  += h;
	Sz  *= decay; Sz  += zh;
	Sx  *= decay; Sx  += xh;
	Sxz *= decay; Sxz.noalias() += xh * z.transpose();
	Sxx *= decay; Sxx += xh.cwiseProduct(x);

	invSzz /= decay;
	// Using Moore-Penrose Rank-1 update
	SZ = invSzz * z;
	DOT = z.dot(SZ);
	DEN = DOT + 1 / h;
	invSzz -= (SZ/DEN) * SZ.transpose();

	invSzz0 /= decay;
	// Using Moore-Penrose Rank-1 update
	SZ = invSzz0 * z;
	DOT = z.dot(SZ);
	DEN = DOT + 1 / h;
	invSzz0 -= (SZ/DEN) * SZ.transpose();

	// Using Determinant Rank-1 update
	detInvSzz0 /= pow(decay,d);
	detInvSzz0 /= (DOT*h + 1.0);

    // Priors decay...
    wPsi *= decay;
    wNu *= decay;
    wSigma *= decay;
    wLambda *= decay;
}

EXPERT_CLASS_TEMPLATE
void FAST_LINEAR_EXPERT_base::m_step()
{
	Z SZ;
	Scal DEN;
	Scal ShSigma = Sh + wSigma, ShNu = Sh + wNu;
    Z meanZ = Sz / Sh;
    X meanX = Sx / Sh;

    if( wNu == 0.0 )
        Nu = meanZ;
    else
        Nu = (Sz + wNu*Nu0) / ShNu;

	// Using Moore-Penrose Rank-1 downdate
	SZ = invSzz0 * Nu;
	DEN = Nu.dot(SZ) - 1.0/ShNu;
	invSigma = (invSzz0 - (SZ/DEN) * SZ.transpose()) * ShSigma;
	// Using Determinant Rank-1 downdate
	sqrtDetInvSigma = sqrt( -detInvSzz0 * ( pow(ShSigma,d) / ShNu / DEN) );


	// Using Moore-Penrose Rank-1 update
	SZ = invSzz * Sz;
	DEN = Sz.dot(SZ) - Sh;
	varLambda = invSzz - (SZ/DEN) * SZ.transpose();

    XZ xz = Sxz;
    xz.noalias() -= meanX*Sz.transpose();
    Lambda.noalias() = xz * varLambda;
    Mu = meanX;
    Mu.noalias() += Lambda * (Nu - meanZ);

    Psi = Sxx;
//Psi += wPsi * mixture->getPsi();
    Psi.noalias() -= meanX.cwiseProduct(Sx) + xz.cwiseProduct(Lambda).rowwise().sum();
    Psi /= (wPsi + Sh + 2.0 );
    invPsi = Psi.cwiseInverse();
    sqrtDetInvPsi = sqrt(invPsi.prod());

    recompute = true;
}

/*
 * FastLinearExpert Display
 */
EXPERT_CLASS_TEMPLATE
void FAST_LINEAR_EXPERT_base::modelDisplay(std::ostream &out) const
{
    out << EXPAND_N(Nu) << std::endl;
    out << EXPAND_N(invSigma) << std::endl;
    out << EXPAND_N(Mu) << std::endl;
    out << EXPAND_N(Lambda) << std::endl;
    out << EXPAND_N(Psi) << std::endl;

    out << EXPAND_N(H) << std::endl;
    out << EXPAND_N(Sh) << std::endl;
    out << EXPAND_N(Sz) << std::endl;
    out << EXPAND_N(Sx) << std::endl;
    out << EXPAND_N(Sxz) << std::endl;
    out << EXPAND_N(invSzz) << std::endl;
    out << EXPAND_N(invSzz0) << std::endl;
    out << EXPAND_N(detInvSzz0) << std::endl;
    out << EXPAND_N(Sxx) << std::endl;
    out << EXPAND_N(Nu0) << std::endl;
//
//    out << EXPAND(wPsi) << std::endl;
//    out << EXPAND(wNu) << std::endl;
//    out << EXPAND(wSigma) << std::endl;
//    out << EXPAND(wLambda) << std::endl;
//    out << EXPAND(alpha) << std::endl;
//    out << EXPAND(mixture->getPsi()) << std::endl;
//    out << EXPAND(mixture->getSigma()) << std::endl;
}



/*
 * LinearExpert Inline members
 */
EXPERT_CLASS_TEMPLATE
template<typename Archive>
void LINEAR_EXPERT_base::serialize(Archive & ar, const unsigned int version)
{
    ar & mixture;

    ar & H;
    ar & Sh;
    ar & Sz;
    ar & Sx;
    ar & Sxz;
    ar & Szz;
    ar & Sxx;

    ar & Nu0;

    ar & alpha;
    Scal wPsiInv = 1.0/wPsi;
    ar & wPsiInv;
    wPsi = 1.0/wPsiInv;
    ar & wNu;
    ar & wSigma;
    ar & wLambda;

    ar & Nu;
    ar & Mu;
    ar & Lambda;
    ar & invSigma;
    ar & Psi;
    ar & invPsi;
    ar & sqrtDetInvSigma;
    ar & varLambda;
}

EXPERT_CLASS_TEMPLATE
template<typename Archive>
void FAST_LINEAR_EXPERT_base::serialize(Archive & ar, const unsigned int version)
{
    ar & H;
    ar & Sh;
    ar & Sz;
    ar & Sx;
    ar & Sxz;
    ar & Sxx;
    ar & invSzz;
    ar & invSzz0;
    ar & detInvSzz0;

    ar & Nu0;

    ar & alpha;
    Scal wPsiInv = 1.0/wPsi;
    ar & wPsiInv;
    wPsi = 1.0/wPsiInv;
    ar & wNu;
    ar & wSigma;
    ar & wLambda;

    ar & Nu;
    ar & Mu;
    ar & Lambda;
    ar & invSigma;
    ar & Psi;
    ar & invPsi;
    ar & sqrtDetInvSigma;
    ar & varLambda;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::get_h() const
{
    return h;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::get_p_z() const
{
    return p_z_T;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::get_p_x() const
{
    return p_x_Norm;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::get_p_zx() const
{
    return p_zx;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::get_rbf_zx() const
{
    return rbf_zx;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::X const &LINEAR_EXPERT_base::getPredX() const
{
    return pred_x;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::X const &LINEAR_EXPERT_base::getPredXVar() const
{
    return Psi;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::X const &LINEAR_EXPERT_base::getPredXInvVar() const
{
    return invPsi;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::getGamma() const
{
    return gamma;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::Z const &LINEAR_EXPERT_base::get_dGamma() const
{
    return dGamma;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::Z const &LINEAR_EXPERT_base::getKappa() const
{
    return kappa;
}

EXPERT_CLASS_TEMPLATE
Scal LINEAR_EXPERT_base::getNewGamma() const
{
    return newGamma;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::Z const &LINEAR_EXPERT_base::get_dNewGamma() const
{
    return dNewGamma;
}


EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::Z const &LINEAR_EXPERT_base::getPredZ() const
{
    return pred_z;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::ZZ const &LINEAR_EXPERT_base::getPredZVar() const
{
    return pred_z_var;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::ZZ const &LINEAR_EXPERT_base::getPredZInvVar() const
{
    return pred_z_invVar;
}

EXPERT_CLASS_TEMPLATE
IMLE_TYPENAME LINEAR_EXPERT_base::ZZ const & LINEAR_EXPERT_base::getUncertFactorPredZ() const
{
    return pred_z_var_factor;
}
