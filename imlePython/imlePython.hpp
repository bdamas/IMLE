#ifndef IMLE_PYTHON_H
#define IMLE_PYTHON_H

#include <boost/python.hpp>

#define IMLE_NO_TEMPLATES

#include "imle.hpp"

// namespace boost::python bp;

struct ImleParam: IMLE::Param {
    ImleParam(int d, int D);

    Eigen::VectorXd getPsi0();
    void setPsi0(const Eigen::VectorXd &newPsi0);
};

class ImlePython: public IMLE {
public:
    ImlePython(int d, int D, const ImleParam &param);

    void reset();

    Eigen::VectorXd predict(const Eigen::VectorXd &z);
    Eigen::VectorXd predictStrongest(const Eigen::VectorXd &z);

    Eigen::VectorXd getPrediction();
    Eigen::VectorXd getPredictionVar();
    Eigen::VectorXd getPredictionJacobian();
    Eigen::VectorXd getPredictionErrorReduction();
    Eigen::VectorXd getPredictionErrorReductionDerivative();

    boost::python::list getMultiplePredictions();
    boost::python::list getMultiplePredictionsVar();
    boost::python::list getMultiplePredictionsWeight();
    boost::python::list getMultiplePredictionsJacobian();
    boost::python::list getMultiplePredictionErrorReduction();
    boost::python::list getMultiplePredictionErrorReductionDerivative();

    boost::python::list getInversePredictions();
    boost::python::list getInversePredictionsVar();
    boost::python::list getInversePredictionsWeight();

    Eigen::VectorXd getSigma();
    Eigen::VectorXd getPsi();
};

#endif
