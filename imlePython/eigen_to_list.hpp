#ifndef EIGEN_TO_LISTH_H
#define EIGEN_TO_LISTH_H

#include <boost/python.hpp>
#include <eigen3/Eigen/Core>
#include <vector>

using namespace Eigen;

VectorXd listToVectorXd(const boost::python::list &x);
MatrixXd listToMatrixXd(const boost::python::list &x);

boost::python::list arrayToList(const std::vector<double> &x);
boost::python::list vectorXdToList(const VectorXd &x);
boost::python::list matrixXdToList(const MatrixXd &x);
boost::python::list arrayVectorXdToList(const std::vector<VectorXd> &x);
boost::python::list arrayMatrixXdToList(const std::vector<MatrixXd> &x);

#endif
