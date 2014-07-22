#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

#include <yarp/sig/all.h>
#include <yarp/os/all.h>

using namespace std;
using namespace yarp::os;

#define OUT_PORT "/queryData1/rpc"

#define SAMPLE_TIME 0.5

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

    v = rf.find("outputDim");
    if( v.isNull() )
    {
        cout << "Warning: no output dimension provided. Using default outputDim = 1." << endl;
        cout << "\tSuggestion: use --outputDim X, where X is he desired output dimension," << endl;
        cout << "\twhen calling this module." << endl;
        outputDim = 1;
    }
    else
        outputDim = v.asInt();


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
        Bottle &q = query.addList();
        q.addDouble(z);
        for(int i = 1; i < inputDim; i++)
            q.addDouble(0.0);

        if( outPort.write(query,response) )
        {
//            cout << response.toString().c_str() << endl;
            double xPred = response.get(0).asList()->findGroup("Prediction").get(1).asDouble();
            cout << "Forward Prediction:" << endl;
            cout << "\t cos(" << z << ") = " << cos(z) << endl;
            cout << "\tpred(" << z << ") = " << xPred << "  (error = " << cos(z)-xPred << ")" << endl;
        }

        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("MultiValued");
        Bottle &q2 = query.addList();
        q2.addDouble(z);
        for(int i = 1; i < inputDim; i++)
            q2.addDouble(0.0);
        vector<double> sol;
        if( outPort.write(query,response) )
        {
            for( int k = 0; k < response.size(); k++ )
                sol.push_back(response.get(k).asList()->findGroup("Prediction").get(1).asDouble());

            cout << "Multiple Prediction:" << endl;
            cout << "\t cos(" << z << ") = " << cos(z) << endl;
            cout << "\tpred(" << z << ") = ( ";
            for( int k = 0; k < sol.size(); k++ )
                cout << sol[k] << ", ";
            cout << ")" << endl;
        }

        double x = 2.0*Random::uniform()-1.0;
        double inv_x = acos(x);
        query.clear(); response.clear();
        query.addString("Predict");
        query.addString("Inverse");
        Bottle &q3 = query.addList();
        q3.addDouble(x);
        for(int i = 1; i < outputDim; i++)
            q3.addDouble(0.0);
        vector<double> invSol;
        if( outPort.write(query,response) )
        {
//            cout << "QUERY\n" << query.toString() << endl;
//            cout << "RESPONSE\n" << response.toString() << endl;
            for( int k = 0; k < response.size(); k++ )
                invSol.push_back(response.get(k).asList()->findGroup("Prediction").get(1).asDouble());
            sort(invSol.begin(), invSol.end() );

            cout << "Inverse Prediction:" << endl;
            cout << "\tinvCos(" << x << ") = ( " << inv_x << ", " << 2.0*M_PI-inv_x << ", " << 2.0*M_PI+inv_x << ", " << 4.0*M_PI-inv_x << " )" << endl;
            cout << "\t  pred(" << x << ") = ( ";
            for( int k = 0; k < invSol.size(); k++ )
                cout << invSol[k] << ", ";
            cout << ")" << endl;
        }

        Time::delay(SAMPLE_TIME);
    }

    outPort.close();
}

