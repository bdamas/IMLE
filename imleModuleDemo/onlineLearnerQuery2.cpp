#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include <yarp/sig/all.h>
#include <yarp/os/all.h>

using namespace std;
using namespace yarp::os;

#define OUT_PORT "/queryData2/rpc"

#define SAMPLE_TIME 0.5

#define NOISE_Z 0.02
#define NOISE_X 0.1


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

    RpcClient outPort;
    if (!outPort.open(OUT_PORT))
    {
        cerr << "Failed to create ports.\n" << "Maybe you need to start a nameserver (run 'yarpserver')\n";
        return 1;
    }

    while( true )
    {
        Bottle query, response;

        query.addString("Predict");
        double z = Random::uniform() * 4*M_PI;
        double x = 2.0*Random::uniform()-1.0;
        {   Bottle &q = query.addList();
            q.addDouble(z);
            for(int i = 1; i < inputDim; i++)
                q.addDouble(0.0);
        }

        if( outPort.write(query,response) )
        {
            cout << query.toString().c_str() << " ===> \n\t";
            cout << response.toString().c_str();
            cout << endl << endl;
        }

        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("SingleValued");
        {   Bottle &q = query.addList();
            q.addDouble(z);
            for(int i = 1; i < inputDim; i++)
                q.addDouble(0.0);
        }
        if( outPort.write(query,response) )
        {
            cout << query.toString().c_str() << " ===> \n\t";
            cout << response.toString().c_str();
            cout << endl << endl;
        }

        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("SingleValued");
        {   Bottle &q = query.addList();
            q.addDouble(z);
            for(int i = 1; i < inputDim; i++)
                q.addDouble(0.0);
        }
        query.addString("WithJacobian");
        if( outPort.write(query,response) )
        {
            cout << query.toString().c_str() << " ===> \n\t";
            cout << response.toString().c_str();
            cout << endl << endl;
        }

        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("Strongest");
        {   Bottle &q = query.addList();
            q.addDouble(z);
            for(int i = 1; i < inputDim; i++)
                q.addDouble(0.0);
        }
        query.addString("WithJacobian");
        if( outPort.write(query,response) )
        {
            cout << query.toString().c_str() << " ===> \n\t";
            cout << response.toString().c_str();
            cout << endl << endl;
        }

        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("MultiValued");
        {   Bottle &q = query.addList();
            q.addDouble(z);
            for(int i = 1; i < inputDim; i++)
                q.addDouble(0.0);
        }
        query.addString("WithJacobian");
        if( outPort.write(query,response) )
        {
            cout << query.toString().c_str() << " ===> \n\t";
            cout << response.toString().c_str();
            cout << endl << endl;
        }
        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("Inverse");
        {   Bottle &q = query.addList();
            q.addDouble(x);
            for(int i = 1; i < outputDim; i++)
                q.addDouble(0.0);
        }
        if( outPort.write(query,response) )
        {
            cout << query.toString().c_str() << " ===> \n\t";
            cout << response.toString().c_str();
            cout << endl << endl;
        }

        Time::delay(SAMPLE_TIME);
    }

    outPort.close();
}


