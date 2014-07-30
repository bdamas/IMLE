#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <iostream>
#include <boost/random.hpp>

#include "imle.hpp"

using namespace std;


#include "imleDemo_common.hpp"

// Demo1: Basics
void demo1()
{
    const int d = 1;    //Input dimension
    const int D = 1;    //Output dimension

    // Default constructor:
    cout << "\t --- Default constructor: ---" << endl;
    IMLE<d,D> imleObj;

    //      IMLE Parameters
    IMLE<d,D>::Param param;
    param.Psi0 = IMLE<d,D>::X::Constant(0.01);
    param.sigma0 = 2.0;
    param.wSigma = 2.0;
    param.wPsi = 20.0;
    param.multiValuedSignificance = 0.95;
    param.p0 = 0.3;

    imleObj.setParameters(param);

    // Training and testing
    cos1D_train(imleObj);
    cos1D_display(imleObj);
    cos1D_predict(imleObj);
}

// Demo2: save, load, parameters...
void demo2()
{
    const int d = 1;    //Input dimension
    const int D = 1;    //Output dimension

    //      IMLE Parameters
    IMLE<d,D>::Param param;
    param.Psi0 = IMLE<d,D>::X::Constant(0.01);
    param.sigma0 = 2.0;
    param.wSigma = 2.0;
    param.wPsi = 20.0;
    param.multiValuedSignificance = 0.95;
    param.p0 = 0.3;

    // Param constructor:
    cout << "\t --- Param constructor: ---" << endl;
    IMLE<d,D> imleObj(param);

    // Default Param File:
    IMLE<d,D>::createDefaultParamFile("cos1D.xml");

    cos1D_testSaveLoadAndParamHandling(imleObj );
}

// Demo3: Multi-valued regression
void demo3()
{
    const int d = 1;    //Input dimension
    const int D = 1;    //Output dimension

    //      IMLE Parameters
    IMLE<d,D>::Param param;
    param.Psi0 = IMLE<d,D>::X::Constant(0.01);
    param.sigma0 = 2.0;
    param.wSigma = 2.0;
    param.wPsi = 20.0;
    param.multiValuedSignificance = 0.95;
    param.p0 = 0.3;

    // IMLE object
    IMLE<d,D> imleObj(param);

    doubleCos1D_train( imleObj );
    doubleCos1D_eval( imleObj );
}

int main(int argc, char **argv)
{
    demo1();
    demo2();
    demo3();

    cout << "\n\t --- IMLE demo ended! ---" << endl;
    return 0;
}
