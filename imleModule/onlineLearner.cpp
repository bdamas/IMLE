#include "onlineLearner.hpp"

#include <sstream>
#include <boost/algorithm/string.hpp>
#include <string>

using namespace std;

using namespace yarp;
using namespace yarp::os;


/******************************************************************************/
/**************************  OnlineLearnerModule  *****************************/
/******************************************************************************/


void OnlineLearnerModule::parseIMLEConfigFile( yarp::os::Property &learnerProperties )
{
    Bottle bottle;
    Value *vp;
	int inputDim = -1, outputDim = -1;

#ifdef IMLE_NO_TEMPLATES
	bottle = learnerProperties.findGroup("Dimensions");
	if( bottle.check("inputDim", vp ) && vp->isInt() && vp->asInt() > 0 )
		inputDim = vp->asInt();
	if( bottle.check("outputDim", vp ) && vp->isInt() && vp->asInt() > 0 )
		outputDim = vp->asInt();

	if( inputDim <= 0 )
	{
		cerr << "IMLE Module: configuration file does not have input dimension!" << endl;
		cerr << "******  USING d = 1 !!  ********" <<endl;
		inputDim = 1;
	}
	if( outputDim <= 0 )
	{
		cerr << "IMLE Module: configuration file does not have output dimension!" << endl;
		cerr << "******  USING D = 1 !!  ********" <<endl;
		outputDim = 1;
	}

	learner = new LearnerMachine( inputDim, outputDim );
#else
	inputDim = INPUT_DIM;
	outputDim = OUTPUT_DIM;
	learner = new LearnerMachine;
#endif


    bottle = learnerProperties.findGroup("General");
    if( bottle.check("loadFromFile", vp ) && vp->isString() && learner->load( vp->asString().c_str() ) )
    {
        cout << getName() << ": loaded previous saved " << vp->asString().c_str() << endl;
        params = learner->getParameters();
    }
    if( bottle.check("defaultSave", vp ) && vp->isString()  && vp->asInt() != -1 )
        params.defaultSave = vp->asString().c_str();
    if( bottle.check("saveOnExit", vp ) && vp->isInt() && vp->asInt() != -1 )
        params.saveOnExit = vp->asInt();

    bottle = learnerProperties.findGroup("Learning");
    if( bottle.check("accelerated", vp ) && vp->isInt() && vp->asInt() != -1 )
        params.accelerated = vp->asInt();
    if( bottle.check("alpha", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.alpha = vp->asDouble();

    if( bottle.check("Psi0", vp ) && vp->asInt() != -1 )
        if( (vp->isDouble() || vp->isInt()) )
			params.Psi0 = LearnerMachine::X::Constant(outputDim, vp->asDouble() );
        else if( vp->isList() && vp->asList()->size() == outputDim )
            for(int i = 0; i < outputDim; i++ )
                if( vp->asList()->get(i).isDouble() )
                    params.Psi0[i] = vp->asList()->get(i).asDouble();
    if( bottle.check("Sigma0", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.sigma0 = vp->asDouble();
    if( bottle.check("wPsi0", vp )  && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.wpsi = vp->asDouble();
    if( bottle.check("wSigma0", vp )  && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.wsigma = vp->asDouble();

    if( bottle.check("wPsi", vp )  && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.wPsi = vp->asDouble();
    if( bottle.check("wSigma", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.wSigma = vp->asDouble();
    if( bottle.check("wNu", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.wNu = vp->asDouble();
    if( bottle.check("wLambda", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.wLambda = vp->asDouble();

    if( bottle.check("p0", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.p0 = vp->asDouble();
    if( bottle.check("nOutliers", vp ) && vp->isInt() && vp->asInt() != -1 )
        params.nOutliers = vp->asInt();

    bottle = learnerProperties.findGroup("Prediction");
    if( bottle.check("multiValuedSignificance", vp ) && (vp->isDouble() || vp->isInt()) && vp->asInt() != -1 )
        params.multiValuedSignificance = vp->asDouble();
    if( bottle.check("nSolMin", vp ) && vp->isInt() && vp->asInt() != -1 )
        params.nSolMin = vp->asInt();
    if( bottle.check("nSolMax", vp ) && vp->isInt() && vp->asInt() != -1 )
        params.nSolMax = vp->asInt();
    if( bottle.check("iterMax", vp ) && vp->isInt() && vp->asInt() != -1 )
        params.iterMax = vp->asInt();
}


bool OnlineLearnerModule::configure(ResourceFinder &rf)
{
    /* Process all parameters from both command-line and .ini file */
    moduleName            = rf.check("name",
                           Value("onlineLearner"),
                           "module name (string)").asString();
    setName(moduleName.c_str());

    handlerPortName         = "/";
    handlerPortName         += getName();

    dataPortName            = "/";
    dataPortName            += getName( rf.check("dataPort",
                               Value("/data:i"),
                               "Data input port (string)").asString()
                               );

    queryPortName           = "/";
    queryPortName           += getName( rf.check("queryPort",
                               Value("/query:rpc"),
                               "Query port (string)").asString()
                               );

    /* IMLE Config File */
    learnerConfigFilename = rf.check("learnerConfig",
                           Value("learner.ini"),
                           "online learner configuration filename (string)").asString().c_str();
    learnerConfigFilename = rf.findFile(learnerConfigFilename.c_str()).c_str();

    Property learnerProperties;
    if (learnerProperties.fromConfigFile(learnerConfigFilename.c_str()) == false)
    {
        cout << getName() << ": unable to read configuration file " << learnerConfigFilename << endl;
        cout << getName() << ":    using default parameters." << endl;
    }
//    else
        parseIMLEConfigFile( learnerProperties );
    learner->setParameters(params);

    /* open ports  */
    if (!dataPort.open(dataPortName.c_str())) {
        cout << getName() << ": unable to open port " << dataPortName << endl;
        return false;  // unable to open; let RFModule know so that it won't run
    }
    if (!queryPort.open(queryPortName.c_str())) {
        cout << getName() << ": unable to open port " << queryPortName << endl;
        return false;  // unable to open; let RFModule know so that it won't run
    }

    if (!handlerPort.open(handlerPortName.c_str())) {
        cout << getName() << ": Unable to open port " << handlerPortName << endl;
        return false;
    }

    attach(handlerPort);                  // attach to port
//    Not recomended by YARP
//    attachTerminal();                     // attach to terminal

    nPoints = 0;
    /* create the threads and pass pointers to the module parameters */
    onlineLearnerThread = new OnlineLearnerThread( &dataPort, &lock, learner, &nPoints );
    onlineLearnerQueryThread = new OnlineLearnerQueryThread( &queryPort, &lock, learner );

    /* now start the thread to do the work */
    onlineLearnerThread->start();
    onlineLearnerQueryThread->start();

    return true ;      // let the RFModule know everything went well
                       // so that it will then run the module
}

bool OnlineLearnerModule::interruptModule()
{
    dataPort.interrupt();
    queryPort.interrupt();
    handlerPort.interrupt();

    return true;
}


bool OnlineLearnerModule::close()
{
    dataPort.close();
    queryPort.close();
    handlerPort.close();

    /* stop the thread */
    onlineLearnerThread->stop();
    onlineLearnerQueryThread->stop();

    delete onlineLearnerThread;
    delete onlineLearnerQueryThread;

    return true;
}

bool OnlineLearnerModule::respond(const Bottle& command, Bottle& reply)
{
    string helpMessage =  string(getName().c_str()) +
                          " commands are: \n" +
                          "help \n" +
                          "quit \n" +
                          "set param <n> ... NOT IMPLEMENTED YET!! \n";

    reply.clear();

    if (command.get(0).asString()=="quit") {
         reply.addString("quitting");
         return false;
    }
    else if (command.get(0).asString()=="help") {
       cout << helpMessage;
       reply.addString("ok");
    }
    else if (command.get(0).asString()=="set") {
//       if (command.get(1).asString()=="thr") {
//          thresholdValue = command.get(2).asInt(); // set parameter value
//          reply.addString("ok");
//       }
       reply.addString("Not implemented yet!");
    }
    return true;
}

/* Called periodically every getPeriod() seconds */

bool OnlineLearnerModule::updateModule()
{
    cout << "IMLE: " << nPoints << " points acquired, M = " << learner->getNumberOfModels() << ", Psi = [" << learner->getPsi().transpose() << "], sigma = [" << learner->getSigma().transpose() << "]." << endl;

    return true;
}



double OnlineLearnerModule::getPeriod()
{
    /* module periodicity (seconds), called implicitly by myModule */
    return 0.5;
}






/******************************************************************************/
/**************************  OnlineLearnerThread  *****************************/
/******************************************************************************/

OnlineLearnerThread::OnlineLearnerThread( BufferedPort< Bottle > *dPort, Semaphore *lck, LearnerMachine *lrn, long *n )
{
    dataPort = dPort;
    lock = lck;
    learner = lrn;
    nPoints = n;

#ifdef IMLE_NO_TEMPLATES
    input_data.resize(lrn->inputDim());
    output_data.resize(lrn->outputDim());
#endif
}

bool OnlineLearnerThread::threadInit()
{
    return true;
}

void OnlineLearnerThread::run()
{
    while ( !isStopping() )
    {
        /*
        * Read the input-output pair from bottle
        */
//        bottle = dataPort->read();
        bottle = dataPort->read(false);
        if( bottle == NULL)
            continue;

        b_input  = bottle->get(0).asList();
        b_output = bottle->get(1).asList();

		for( int i = 0; i < learner->inputDim(); i++ )
            input_data[i] = b_input->get(i).asDouble();

        for( int i = 0; i < learner->outputDim(); i++ )
            output_data[i] = b_output->get(i).asDouble();
//cout << "Got: " << input_data << ", " << output_data << endl;
        /*
        * Pass it to the learner (mutex lock)
        */
        lock->wait();
        learner->update(input_data, output_data);
        lock->post();
        (*nPoints)++;
    }
}


void OnlineLearnerThread::threadRelease()
{
}

/******************************************************************************/
/************************  OnlineLearnerQueryThread  **************************/
/******************************************************************************/

OnlineLearnerQueryThread::OnlineLearnerQueryThread( RpcServer *qPort, Semaphore *lck, LearnerMachine *lrn )
{
    queryPort = qPort;
    lock = lck;
    learner = lrn;
}

bool OnlineLearnerQueryThread::threadInit()
{
    return true;
}


void OnlineLearnerQueryThread::bottleSolutions(QueryType queryType, bool getJacobian)
{
    int dimPred;
    int nRows;
    int nCols;

    response.clear();

    if( queryType == Predict || queryType == PredictStrongest)
    {
        // In case something changes in IMLE data storage...
        dimPred = learner->getPrediction().size();
        nRows = learner->getPredictionVar().rows();
        nCols = learner->getPredictionVar().cols();

        Bottle &sol = response.addList();

        Bottle &pred = sol.addList();
        pred.addString("Prediction");
        for(int j = 0; j < dimPred; j++)
            pred.addDouble( learner->getPrediction()(j) );

        Bottle &predVar = sol.addList();
        predVar.addString("Variance");
        for(int j = 0; j < nRows; j++)
            for(int k = 0; k < nCols; k++)
                predVar.addDouble( learner->getPredictionVar()(j,k) );

        Bottle &predWeight = sol.addList();
        predWeight.addString("Weight");
        predWeight.addDouble( learner->getPredictionWeight() );

        if( getJacobian )
        {
            nRows = learner->getPredictionJacobian().rows();
            nCols = learner->getPredictionJacobian().cols();

            Bottle &predJac = sol.addList();
            predJac.addString("Jacobian");
            for(int j = 0; j < nRows; j++)
                for(int k = 0; k < nCols; k++)
                    predJac.addDouble( learner->getPredictionJacobian()(j,k) );
        }
    }
    else if( queryType == PredictMultiple )
    {
        // In case something changes in IMLE data storage...
        dimPred = learner->getMultiplePredictions()[0].size();
        nRows = learner->getMultiplePredictionsVar()[0].rows();
        nCols = learner->getMultiplePredictionsVar()[0].cols();

        int nSol = learner->getNumberOfSolutionsFound();
        if( nSol == 0 )
            return;

        for( int i = 0; i < nSol; i++)
        {
            Bottle &sol = response.addList();

            Bottle &pred = sol.addList();
            pred.addString("Prediction");
            for(int j = 0; j < dimPred; j++)
                pred.addDouble( learner->getMultiplePredictions()[i](j) );

            Bottle &predVar = sol.addList();
            predVar.addString("Variance");
            for(int j = 0; j < nRows; j++)
                for(int k = 0; k < nCols; k++)
                    predVar.addDouble( learner->getMultiplePredictionsVar()[i](j,k) );

            Bottle &predWeight = sol.addList();
            predWeight.addString("Weight");
            predWeight.addDouble( learner->getMultiplePredictionsWeight()[i] );

            if( getJacobian )
            {
                nRows = learner->getMultiplePredictionsJacobian()[0].rows();
                nCols = learner->getMultiplePredictionsJacobian()[0].cols();

                Bottle &predJac = sol.addList();
                predJac.addString("Jacobian");
                for(int j = 0; j < nRows; j++)
                    for(int k = 0; k < nCols; k++)
                        predJac.addDouble( learner->getMultiplePredictionsJacobian()[i](j,k) );
            }
        }
    }
    else if( queryType == PredictInverse)
    {
        // In case something changes in IMLE data storage...
        dimPred = learner->getInversePredictions()[0].size();
        nRows = learner->getInversePredictionsVar()[0].rows();
        nCols = learner->getInversePredictionsVar()[0].cols();

        int nSol = learner->getNumberOfInverseSolutionsFound();
        if( nSol == 0 )
            return;

        for( int i = 0; i < nSol; i++)
        {
            Bottle &sol = response.addList();

            Bottle &pred = sol.addList();
            pred.addString("Prediction");
            for(int j = 0; j < dimPred; j++)
                pred.addDouble( learner->getInversePredictions()[i](j) );

            Bottle &predVar = sol.addList();
            predVar.addString("Variance");
            for(int j = 0; j < nRows; j++)
                for(int k = 0; k < nCols; k++)
                    predVar.addDouble( learner->getInversePredictionsVar()[i](j,k) );

            Bottle &predWeight = sol.addList();
            predWeight.addString("Weight");
            predWeight.addDouble( learner->getInversePredictionsWeight()[i] );
        }
    }
}

bool OnlineLearnerQueryThread::readVector(Vec &v, int pos, int dims)
{
    if( !query.get(pos).isList() )
        return false;

    Bottle *bp = query.get(pos).asList();
    if( bp->size() != dims )
        return false;

    queryVector.resize(dims);
    for( int i=0; i < dims; i++ )
        queryVector[i] = bp->get(i).asDouble();
    return true;
}

void OnlineLearnerQueryThread::run()
{
    QueryType queryType;

    while ( !isStopping() )
    {
        query.clear();
        queryPort->read(query,true);

        string queryString = query.get(0).asString().c_str();
        boost::to_lower(queryString);

        if( queryString  == "predict" )
        {
            queryString = query.get(1).asString().c_str();
            boost::to_lower(queryString);

            if( query.get(1).isList() )
                if( readVector(queryVector, 1, learner->inputDim()) )
                    queryType = Predict;
                else break;
            else if( queryString == "singlevalued" )
                if( readVector(queryVector, 2, learner->inputDim()) )
                    queryType = Predict;
                else break;
            else if( queryString == "multivalued" )
                if( readVector(queryVector, 2, learner->inputDim()) )
                    queryType = PredictMultiple;
                else break;
            else if( queryString == "strongest" )
                if( readVector(queryVector, 2, learner->inputDim()) )
                    queryType = PredictStrongest;
                else break;
            else if( queryString == "inverse" )
                if( readVector(queryVector, 2, learner->outputDim()) )
                    queryType = PredictInverse;
                else break;

            // Check if the prediction Jacobian is necessary
            queryString = query.get(query.size()-1).asString();
            boost::to_lower(queryString);
            bool getJacobian = (queryString == "withjacobian");


            lock->wait();
            switch (queryType)
            {
                case Predict:
                    learner->predict(queryVector); break;
                case PredictMultiple:
                    learner->predictMultiple(queryVector); break;
                case PredictStrongest:
                    learner->predictStrongest(queryVector); break;
                case PredictInverse:
                    learner->predictInverse(queryVector); break;
            }
            bottleSolutions(queryType, getJacobian);
            lock->post();
        }
        else if( queryString  == "help" )
        {
                string helpString = "\
IMLE Module available commands:\n\
\tPredict x_1 x_2 ... x_INPUT_DIM\n\
\tPredictMultiple x_1 x_2 ... x_INPUT_DIM\n\
\tPredictStrongest x_1 x_2 ... x_INPUT_DIM\n\
\tPredictInverse y_1 y_2 ... y_INPUT_DIM\n\
\tPredictInverseSingle y_1 y_2 ... y_INPUT_DIM\n\
\tPredictInverseStrongest y_1 y_2 ... y_INPUT_DIM\n\
\tHelp\n\
Example:\n\
\t\tRpcClient port; port.open(\"outPort\");  //outPort is connected to onlineLearnerQuery\n\
\t\tyarp.connect(\"outPort\",\"onlineLearner/query:rpc\");\n\
\t\tBottle query,response;\n\
\t\tquery.addString(\"Predict\");\n\
\t\tquery.addDouble(0.1);\n\
\t\tquery.addDouble(1.2);\n\
\t\tport.write(query,response);\n\
\t\tstd::cout << \"IMLE responds with \" << response.toString().c_str() << std::endl;\n";
                response.addString( helpString.c_str() );
        }

        if( response.size() == 0 )
            response.addString("IMLE Module: Malformed query. Try \"Help\" for available commands.");

        queryPort->reply(response);
    }
}




void OnlineLearnerQueryThread::threadRelease()
{
}

