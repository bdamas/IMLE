#include <string>
#include <iostream>
#include <boost/random.hpp>

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

        cos1D_train(imleObj);
        cos1D_display(imleObj);
        cos1D_predict(imleObj);
    }

    cout << "\n\t --- IMLE demo ended! ---" << endl;
    return 0;
}
