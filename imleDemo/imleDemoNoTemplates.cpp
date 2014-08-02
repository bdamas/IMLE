#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <iostream>
#include <boost/random.hpp>

#define IMLE_NO_TEMPLATES
#include "imle.hpp"

using namespace std;


#include "imleDemo_common.hpp"

// Demo1: Basics
void demo1()
{
    int d = 1;    //Input dimension
    int D = 1;    //Output dimension

    // Default constructor:
    cout << "\t --- Default constructor: ---" << endl;
    IMLE imleObj(d,D);

    //      IMLE Parameters
    IMLE::Param param(d,D);
    param.Psi0 = IMLE::X::Constant(D,0.1);
    param.sigma0 = 1.0;
    param.p0 = 0.2;
    param.multiValuedSignificance = 0.9;

    imleObj.setParameters(param);

    cos1D_train(imleObj);
    cos1D_predict(imleObj);
}

// Demo2: save, load, parameters...
void demo2()
{
    int d = 1;    //Input dimension
    int D = 1;    //Output dimension

    //      IMLE Parameters
    IMLE::Param param(d,D);
    param.Psi0 = IMLE::X::Constant(D,0.1);
    param.sigma0 = 1.0;
    param.p0 = 0.2;

    // Param constructor:
    cout << "\t --- Param constructor: ---" << endl;
    IMLE imleObj(d,D,param);

    // Default Param File:
    IMLE::createDefaultParamFile(d,D,"cos1D.xml");

    cos1D_testSaveLoadAndParamHandling(imleObj );
}

// Demo3: Multi-valued regression
void demo3()
{
    int d = 1;    //Input dimension
    int D = 1;    //Output dimension

    //      IMLE Parameters
    IMLE::Param param(d,D);
    param.Psi0 = IMLE::X::Constant(D,0.1);
    param.sigma0 = 1.0;
    param.p0 = 0.3;

    // IMLE object
    IMLE imleObj(d,D,param);

    doubleCos1D_train( imleObj );
    doubleCos1D_eval( imleObj );
}

int main(int argc, char **argv)
{
    demo1();
    demo2();
    demo3();

    cout << "\n\t --- IMLE demo (no templates) ended! ---" << endl;
    return 0;
}

