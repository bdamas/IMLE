#include "imlePython.hpp"
#include "eigen_to_list.hpp"

ImleParam::ImleParam(int d, int D) : IMLE::Param(d, D) {
}

Eigen::VectorXd ImleParam::getPsi0() {
    return Psi0;
}

void ImleParam::setPsi0(const Eigen::VectorXd &newPsi0) {
    printf("newPsi0 = %g", newPsi0[0]);
    Psi0 = newPsi0;
    printf("Psi0 = %g", newPsi0[0]);
}

ImlePython::ImlePython(int d, int D, const ImleParam &param) : IMLE(d, D, param) {
    printf("bite3 : %g \n", param.sigma0);
}

void ImlePython::reset() {
    IMLE::reset();
}

boost::python::list ImlePython::getJointMu(int expert) {
    boost::python::list l;
    for(int i=0; i<d; i++)
        l.append(IMLE::getExperts()[expert].Nu[i]);

    for(int i=0; i<D; i++)
        l.append(IMLE::getExperts()[expert].Mu[i]);

    return l;
}

// boost::python::list ImlePython::getInvSigma(int expert) {
//     IMLE::ZZ A=IMLE::getExperts()[expert].getInvSigma();
//     boost::python::list ll;
// 	for(int i=0; i<d; i++) {
// 	    boost::python::list l;
// 	    for(int j=0; j<d; j++)
//             	l.append(A(i, j));
// 	    ll.append(l);
// 	}
//     return ll;
// }

boost::python::list ImlePython::getLambda(int expert) {
    return matrixXdToList(IMLE::getExperts()[expert].Lambda);
}

// boost::python::list ImlePython::getPsi(int expert) {
//     IMLE::X A=IMLE::getExperts()[expert].getPsi();
//     boost::python::list l;
// 	for(int i=0; i<D; i++) {
// 		l.append(A(i));
// 	}
//     return l;
// }

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

boost::python::list ImlePython::getInvSigma(int expert) {
    return matrixXdToList(IMLE::getExperts()[expert].invSigma);
}

boost::python::list ImlePython::getPsi(int expert) {
    return vectorXdToList(getExperts()[expert].Psi);
}
