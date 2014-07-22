
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

#include <yarp/sig/all.h>
#include <yarp/os/all.h>

using namespace std;
using namespace yarp::os;

#define OUT_PORT "/genData/out"

#define SAMPLE_TIME 0.2

#define NOISE_Z 0.02
#define NOISE_X 0.1

#ifndef M_PI
    #define M_PI       3.14159265358979323846  // Visual Studio was reported not to define M_PI, even when including cmath and defining _USE_MATH_DEFINES...
#endif

int main(int argc, char *argv[])
{
    /* initialize yarp network */

    Network yarp;
    if ( !yarp.checkNetwork() )
        return -1;

    int inputDim;
    int outputDim;
    ResourceFinder rf;
    rf.configure(argc, argv);


    // An odd behaviour happens with yarp: the following code
    //      Value v = rf.find("inputDim");
    //      (...)
    //      v = rf.find("outputDim");
    //
    // results in v.isNull() returning always false even if v.asString() returns an empty string...
    {
        Value v = rf.find("inputDim");
        if( v.isNull() )
        {
            cout << "Warning: no input dimension provided. Using default inputDim = 1." << endl;
            cout << "\tSuggestion: use --inputDim X, where X is he desired input dimension," << endl;
            cout << "\twhen calling this module." << endl;
            inputDim = 1;
        }
        else
            inputDim = v.asInt();
    }

    {
        Value v = rf.find("outputDim");
        if( v.isNull() )
        {
            cout << "Warning: no output dimension provided. Using default outputDim = 1." << endl;
            cout << "\tSuggestion: use --outputDim X, where X is he desired output dimension," << endl;
            cout << "\twhen calling this module." << endl;
            outputDim = 1;
        }
        else
            outputDim = v.asInt();
    }



    BufferedPort<Bottle> outPort;
    if (!outPort.open(OUT_PORT))
    {
        cerr << "Failed to create ports.\n" << "Maybe you need to start a nameserver (run 'yarpserver')\n";
        return 1;
    }

    double z = 0.0, x = 0.0;
    while( true )
    {
        z += Random::normal() * NOISE_Z;
        x = cos(z) + NOISE_X * Random::normal();

        Bottle &out = outPort.prepare();
        out.clear();

        // out -> ( (z) (x) )
        Bottle &zB = out.addList();
        zB.addDouble(z);
        for(int i = 1; i < inputDim; i++)
            zB.addDouble(0.0);

        Bottle &xB = out.addList();
        xB.addDouble(x);
        for(int i = 1; i < outputDim; i++)
            xB.addDouble(0.0);

        cout << "genData: sending " << out.toString().c_str() << endl;
        outPort.write();

        Time::delay(SAMPLE_TIME);

        if( z > 4*M_PI )
            z = 0.0;
        else
            z += 0.1;
    }

    outPort.close();
}


