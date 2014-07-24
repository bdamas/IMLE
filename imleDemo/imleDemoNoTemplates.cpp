#include <string>
#include <iostream>
#include <boost/random.hpp>

#define IMLE_NO_TEMPLATES
#include "imle.hpp"

using namespace std;


#include "imleDemo_common.hpp"


int main(int argc, char **argv)
{
    // Demo PART 1
    {
        const int d = 1;    //Input dimension
        const int D = 1;    //Output dimension

        // Default constructor:
        cout << "\t --- Default constructor: ---" << endl;
        IMLE imleObj(d,D);

        //      IMLE Parameters
        IMLE::Param param(d,D);
        param.Psi0 = IMLE::X::Constant(D,0.01);
        param.sigma0 = 2.0;
        param.wSigma = 2.0;
        param.wPsi = 20.0;
        param.multiValuedSignificance = 0.95;
        param.p0 = 0.3;

        imleObj.setParameters(param);

        cos1D_train(imleObj);
        cos1D_display(imleObj);
        cos1D_predict(imleObj);
    }

    cout << "\n\t --- IMLE demo (no templates) ended! ---" << endl;
    return 0;
}

