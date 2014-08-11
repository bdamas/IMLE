#include "eigen_to_list.hpp"

VectorXd listToVectorXd(const boost::python::list &x) {
    int len = boost::python::len(x);
    VectorXd res(len);
    for (int i=0; i < len; i++) {
        res(i) = boost::python::extract<double>(x[i]);
    }
    return res;
}

MatrixXd listToMatrixXd(const boost::python::list &x) {
    int nrows = boost::python::len(x);
    int ncols = boost::python::len(x[0]);
    MatrixXd res(nrows, ncols);
    for (int i=0; i < nrows; i++) {
        for (int j=0; j < ncols; j++) {
            res(i, j) = boost::python::extract<double>(x[i][j]);
        }
    }
    return res;
}

boost::python::list arrayToList(const std::vector<double> &x) {
    boost::python::list res;

    for (int i=0; i < x.size(); i++) {
        res.append(x[i]);
    }

    return res;
}

boost::python::list vectorXdToList(const VectorXd &x) {
    int len = x.size();
    boost::python::list res;
    for (int i=0; i < len; i++) {
        res.append(x(i));
    }
    return res;
}

boost::python::list matrixXdToList(const MatrixXd &x) {
    int nrows = x.rows();
    int ncols = x.cols();
    boost::python::list res;
    for (int i=0; i < nrows; i++) {
        boost::python::list row;
        for (int j=0; j < ncols; j++) {
            row.append(x(i, j));
        }
        res.append(row);
    }
    return res;
}

boost::python::list arrayVectorXdToList(const std::vector<VectorXd> &x) {
    boost::python::list res;

    for (int i=0; i < x.size(); i++) {
        res.append(vectorXdToList(x[i]));
    }

    return res;
}

boost::python::list arrayMatrixXdToList(const std::vector<MatrixXd> &x) {
    boost::python::list res;

    for (int i=0; i < x.size(); i++) {
        res.append(matrixXdToList(x[i]));
    }

    return res;
}
