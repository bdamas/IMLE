#include <boost/python.hpp>

#include "eigen_to_numpy.hpp"
#include "imlePython.hpp"


using namespace boost::python;


BOOST_PYTHON_MODULE(_imle)
{
    setupEigenToNumpyConverters();

    class_<ImleParam>("ImleParam", init<int, int>())
        .def_readwrite("accelerated", &IMLE::Param::accelerated)
        .def_readwrite("alpha", &IMLE::Param::alpha)

        // .add_property("Psi0", &ImleParam::getPsi0, &ImleParam::setPsi0)
        .def("_get_psi0", &ImleParam::getPsi0)
        .def("_set_psi0", &ImleParam::setPsi0)
        .def_readwrite("sigma0", &IMLE::Param::sigma0)
        .def_readwrite("wsigma", &IMLE::Param::wsigma)
        .def_readwrite("wpsi", &IMLE::Param::wpsi)

        .def_readwrite("wSigma", &IMLE::Param::wSigma)
        .def_readwrite("wNu", &IMLE::Param::wNu)
        .def_readwrite("wLambda", &IMLE::Param::wLambda)
        .def_readwrite("wPsi", &IMLE::Param::wPsi)

        .def_readwrite("nOutliers", &IMLE::Param::nOutliers)
        .def_readwrite("p0", &IMLE::Param::p0)

        .def_readwrite("multiValuedSignificance", &IMLE::Param::multiValuedSignificance)
        .def_readwrite("nSolMin", &IMLE::Param::nSolMin)
        .def_readwrite("nSolMax", &IMLE::Param::nSolMax)
        .def_readwrite("iterMax", &IMLE::Param::iterMax)
    ;

    class_<ImlePython>("Imle", init<int, int, ImleParam>())
        .def("reset", &ImlePython::reset)

        .def("update", &IMLE::update)

        .def("predict", &ImlePython::predict)
        .def("predictStrongest", &ImlePython::predictStrongest)
        .def("predictMultiple", &IMLE::predictMultiple)

        .def("predictInverse", &IMLE::predictInverse)

        .def("getPrediction", &ImlePython::getPrediction)
        .def("getPredictionVar", &ImlePython::getPredictionVar)
        .def("getPredictionWeight", &IMLE::getPredictionWeight)
        .def("getPredictionJacobian", &ImlePython::getPredictionJacobian)
        .def("getPredictionErrorReduction", &ImlePython::getPredictionErrorReduction)
        .def("getPredictionErrorReductionDerivative", &ImlePython::getPredictionErrorReductionDerivative)

        .def("getMultiplePredictions", &ImlePython::getMultiplePredictions)
        .def("getMultiplePredictionsVar", &ImlePython::getMultiplePredictionsVar)
        .def("getMultiplePredictionsWeight", &ImlePython::getMultiplePredictionsWeight)
        .def("getNumberOfSolutionsFound", &IMLE::getNumberOfSolutionsFound)
        .def("getMultiplePredictionsJacobian", &ImlePython::getMultiplePredictionsJacobian)
        .def("getMultiplePredictionErrorReduction", &ImlePython::getMultiplePredictionErrorReduction)
        .def("getMultiplePredictionErrorReductionDerivative", &ImlePython::getMultiplePredictionErrorReductionDerivative)

        .def("getInversePredictions", &ImlePython::getInversePredictions)
        .def("getInversePredictionsVar", &ImlePython::getInversePredictionsVar)
        .def("getInversePredictionsWeight", &ImlePython::getInversePredictionsWeight)
        .def("getNumberOfInverseSolutionsFound", &IMLE::getNumberOfInverseSolutionsFound)

        .def("inputDim", &IMLE::inputDim)
        .def("outputDim", &IMLE::outputDim)

        .def("save", &IMLE::save)
        .def("load", &IMLE::load)

        .def("getNumberOfExperts", &IMLE::getNumberOfExperts)
        .def("getInvSigma", &ImlePython::getInvSigma)
        .def("getPsi", &ImlePython::getPsi)
        .def("getLambda", &ImlePython::getLambda)
        .def("getJointMu", &ImlePython::getJointMu)
    ;
}
