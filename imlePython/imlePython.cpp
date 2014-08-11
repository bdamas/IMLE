#include "imlePython.hpp"
#include "eigen_to_list.hpp"

ImleParam::ImleParam(int d, int D) : IMLE::Param(d, D) {
}

Eigen::VectorXd ImleParam::getPsi0() {
    return Psi0;
}

void ImleParam::setPsi0(const Eigen::VectorXd &newPsi0) {
    Psi0 = newPsi0;
}

ImlePython::ImlePython(int d, int D, const ImleParam &param) : IMLE(d, D, param) {
}

void ImlePython::reset() {
    IMLE::reset();
}

Eigen::VectorXd ImlePython::predict(const Eigen::VectorXd &z) {
    return IMLE::predict(z);
}

Eigen::VectorXd ImlePython::predictStrongest(const Eigen::VectorXd &z) {
    return IMLE::predictStrongest(z);
}

Eigen::VectorXd ImlePython::getPrediction() {
    return IMLE::getPrediction();
}

Eigen::VectorXd ImlePython::getPredictionVar() {
    return IMLE::getPredictionVar();
}

Eigen::VectorXd ImlePython::getPredictionJacobian() {
    return IMLE::getPredictionJacobian();
}

Eigen::VectorXd ImlePython::getPredictionErrorReduction() {
    return IMLE::getPredictionErrorReduction();
}

Eigen::VectorXd ImlePython::getPredictionErrorReductionDerivative() {
    return IMLE::getPredictionErrorReductionDerivative();
}

boost::python::list ImlePython::getMultiplePredictions() {
    return arrayVectorXdToList(IMLE::getMultiplePredictions());
}

boost::python::list ImlePython::getMultiplePredictionsVar() {
    return arrayVectorXdToList(IMLE::getMultiplePredictionsVar());
}

boost::python::list ImlePython::getMultiplePredictionsWeight() {
    return arrayToList(IMLE::getMultiplePredictionsWeight());
}

boost::python::list ImlePython::getMultiplePredictionsJacobian() {
    return arrayMatrixXdToList(IMLE::getMultiplePredictionsJacobian());
}

boost::python::list ImlePython::getMultiplePredictionErrorReduction() {
    return arrayVectorXdToList(IMLE::getMultiplePredictionErrorReduction());
}

boost::python::list ImlePython::getMultiplePredictionErrorReductionDerivative() {
    return arrayMatrixXdToList(IMLE::getMultiplePredictionErrorReductionDerivative());
}

boost::python::list ImlePython::getInversePredictions() {
    return arrayVectorXdToList(IMLE::getInversePredictions());
}

boost::python::list ImlePython::getInversePredictionsVar() {
    return arrayMatrixXdToList(IMLE::getInversePredictionsVar());
}

boost::python::list ImlePython::getInversePredictionsWeight() {
    return arrayToList(IMLE::getInversePredictionsWeight());
}

Eigen::VectorXd ImlePython::getSigma() {
    return IMLE::getSigma();
}

Eigen::VectorXd ImlePython::getPsi() {
    return IMLE::getPsi();
}
