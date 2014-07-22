#include "onlineLearner.hpp"

using namespace yarp;
using namespace yarp::os;
using namespace std;

int main(int argc, char * argv[])
{
    /* initialize yarp network */

    Network yarp;
    if ( !yarp.checkNetwork() )
        return -1;


    /* prepare and configure the resource finder */

    ResourceFinder rf;
    rf.setVerbose(true);
    rf.setDefaultConfigFile("onlineLearnerModule.ini"); //overridden by --from parameter
//    rf.setDefaultContext("onlineLearnerModule/conf");   //overridden by --context parameter
    rf.configure("ICUB_ROOT", argc, argv);

    /* create your module */

    OnlineLearnerModule onlineLearnerModule;

    /* run the module: runModule() calls configure first and, if successful, it then runs */

    onlineLearnerModule.runModule(rf);

    return 0;
}

