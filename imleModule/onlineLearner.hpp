#ifndef __ICUB_ONLINELEARNER_MODULE_H__
#define __ICUB_ONLINELEARNER_MODULE_H__

#include <iostream>
#include <string>
#include <vector>

#include <yarp/sig/all.h>
#include <yarp/os/all.h>

#include "imle.hpp"

//#define INPUT_DIM 1
//#define OUTPUT_DIM 1

#ifdef IMLE_NO_TEMPLATES
    typedef IMLE LearnerMachine;
#else
    typedef IMLE<INPUT_DIM,OUTPUT_DIM> LearnerMachine;
#endif


/******************************************************************************/
/**************************  OnlineLearnerThread  *****************************/
/******************************************************************************/

class OnlineLearnerThread : public yarp::os::Thread
{
    /* class variables */
    yarp::os::Bottle *bottle, *b_input, *b_output;
    LearnerMachine::Z input_data;
    LearnerMachine::X output_data;

    /* thread parameters: they are pointers so that they refer to the original variables in myModule */
    yarp::os::BufferedPort< yarp::os::Bottle > *dataPort;
    LearnerMachine *learner;
    yarp::os::Semaphore *lock;
    long *nPoints;

public:
    /* class methods */
    OnlineLearnerThread( yarp::os::BufferedPort< yarp::os::Bottle > *dPort, yarp::os::Semaphore *lck, LearnerMachine *lrn, long *n );
    bool threadInit();
    void threadRelease();
    void run();
};


/******************************************************************************/
/************************  OnlineLearnerQueryThread  **************************/
/******************************************************************************/

class OnlineLearnerQueryThread : public yarp::os::Thread
{
    /* class variables */
    yarp::os::Bottle query, response;

    enum QueryType {Predict, PredictMultiple, PredictStrongest, PredictInverse} queryType;

    Vec queryVector;
//    LearnerMachine::Z z;
//    LearnerMachine::X x;

//    void bottleSolutions(bool forward, bool singlevalued);
    void bottleSolutions(QueryType queryType, bool getJacobian);
    bool readVector(Vec &v, int pos, int dims);

    /* thread parameters: they are pointers so that they refer to the original variables in myModule */
    yarp::os::RpcServer *queryPort;
    LearnerMachine *learner;
    yarp::os::Semaphore *lock;

public:
    /* class methods */
    OnlineLearnerQueryThread( yarp::os::RpcServer *qPort, yarp::os::Semaphore *lck, LearnerMachine *lrn );
    bool threadInit();
    void threadRelease();
    void run();
};


/******************************************************************************/
/**************************  OnlineLearnerModule  *****************************/
/******************************************************************************/

class OnlineLearnerModule : public yarp::os::RFModule
{
    /* module parameters */

    std::string moduleName;
    std::string handlerPortName;

    std::string dataPortName;
    std::string queryPortName;

    std::string learnerConfigFilename;
    void parseIMLEConfigFile( yarp::os::Property &learnerProperties );

    /* class variables */

    yarp::os::BufferedPort< yarp::os::Bottle > dataPort;
    yarp::os::RpcServer queryPort;
    yarp::os::Port handlerPort;                           // a port to handle messages

    yarp::os::Semaphore lock;
    long nPoints;

    LearnerMachine *learner;

   /* pointer to a new thread to be created and started in configure() and stopped in close() */

    OnlineLearnerThread *onlineLearnerThread;
    OnlineLearnerQueryThread *onlineLearnerQueryThread;

public:

    bool configure(yarp::os::ResourceFinder &rf); // configure all the module parameters and return true if successful
    bool interruptModule();                       // interrupt, e.g., the ports
    bool close();                                 // close and shut down the module
    bool respond(const yarp::os::Bottle& command, yarp::os::Bottle& reply);
    double getPeriod();
    bool updateModule();
};



#endif


