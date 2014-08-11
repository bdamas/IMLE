#include <Eigen/Core>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <iostream>

namespace boopy
{
  namespace bp = boost::python;

  template <typename SCALAR>  struct NumpyEquivalentType {};
  template <> struct NumpyEquivalentType<double>  { enum { type_code = NPY_DOUBLE };};
  template <> struct NumpyEquivalentType<int>     { enum { type_code = NPY_INT    };};
  template <> struct NumpyEquivalentType<float>   { enum { type_code = NPY_FLOAT  };};

  /* --- TO PYTHON -------------------------------------------------------------- */
  template< typename MatType >
  struct EigenMatrix_to_python_matrix
  {
    static PyObject* convert(MatType const& mat)
    {
      typedef typename MatType::Scalar T;
      const int R  = mat.rows(), C = mat.cols();

      npy_intp shape[2] = { R,C };
      PyArrayObject* pyArray = (PyArrayObject*)
	PyArray_SimpleNew(2, shape,
			  NumpyEquivalentType<T>::type_code);

      T* pyData = (T*)PyArray_DATA(pyArray);
      Eigen::Map< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > pyMatrix(pyData,R,C);
      pyMatrix = mat;

      return (PyObject*)pyArray;
    }
  };

  /* --- FROM PYTHON ------------------------------------------------------------ */
  template<typename MatType>
  struct EigenMatrix_from_python_array
  {

    EigenMatrix_from_python_array()
    {
      bp::converter::registry
	::push_back(&convertible,
		    &construct,
		    bp::type_id<MatType>());
    }

    // Determine if obj_ptr can be converted in a Eigenvec
    static void* convertible(PyObject* obj_ptr)
    {
      typedef typename MatType::Scalar T;

      if (!PyArray_Check(obj_ptr)) return 0;

    //   std::cout << "Until here ok.   ndim = " << PyArray_NDIM(obj_ptr) << " isvec " << MatType::IsVectorAtCompileTime << std::endl;
      if (PyArray_NDIM(obj_ptr) != 2)
	if ( (PyArray_NDIM(obj_ptr) !=1) || (! MatType::IsVectorAtCompileTime) )
	  return 0;
    //   std::cout << "Until here ok." << std::endl;

      if (PyArray_ObjectType(obj_ptr, 0) != NumpyEquivalentType<T>::type_code)
	return 0;

      if (!(PyArray_FLAGS(obj_ptr) & NPY_ALIGNED))
	{
	//   std::cerr << "NPY non-aligned matrices are not implemented." << std::endl;
	  return 0;
	}

      return obj_ptr;
    }

    // Convert obj_ptr into a Eigenvec
    static void construct(PyObject* pyObj,
			  bp::converter::rvalue_from_python_stage1_data* memory)
    {
      typedef typename MatType::Scalar T;
      using namespace Eigen;

    //   std::cout << "Until here ok. Constructing..." << std::endl;
      PyArrayObject * pyArray = reinterpret_cast<PyArrayObject*>(pyObj);

      if ( PyArray_NDIM(pyArray) == 2 )
      {
	int R = MatType::RowsAtCompileTime;
	int C = MatType::ColsAtCompileTime;
	if (R == Eigen::Dynamic) R = PyArray_DIMS(pyArray)[0];
	else	               assert(PyArray_DIMS(pyArray)[0]==R);

	if (C == Eigen::Dynamic) C = PyArray_DIMS(pyArray)[1];
	else	               assert(PyArray_DIMS(pyArray)[1]==C);

	T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));

	int itemsize = PyArray_ITEMSIZE(pyArray);
	int stride1 = PyArray_STRIDE(pyArray, 0) / itemsize;
	int stride2 = PyArray_STRIDE(pyArray, 1) / itemsize;
	// std::cout << "STRIDE = " << stride1 << " x " << stride2 << std::endl;
	Eigen::Map<MatType,0,Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> >
	  pyMap( pyData, R,C, Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic>(stride2,stride1) );
	// std::cout << "Map = " << pyMap << std::endl;

	void* storage = ((bp::converter::rvalue_from_python_storage<MatType>*)
			 (memory))->storage.bytes;
	MatType & mat = * new (storage) MatType(R,C);
	mat = pyMap;

	memory->convertible = storage;
      }
    else
      {
	int R = MatType::MaxSizeAtCompileTime, C=1;
	if(R==Eigen::Dynamic) R =  PyArray_DIMS(pyArray)[0];
	else                  assert(PyArray_DIMS(pyArray)[0]==R);

	T* pyData = reinterpret_cast<T*>(PyArray_DATA(pyArray));

	int itemsize = PyArray_ITEMSIZE(pyArray);
	int stride = PyArray_STRIDE(pyArray, 0) / itemsize;
	Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> s(stride,0);
	Eigen::Map<MatType,0,Eigen::InnerStride<Eigen::Dynamic> >
	  pyMap( pyData, R, 1, Eigen::InnerStride<Eigen::Dynamic>(stride) );
	// std::cout << "Map = " << pyMap << std::endl;

	void* storage = ((bp::converter::rvalue_from_python_storage<MatType>*)
			 (memory))->storage.bytes;
	MatType & mat = * new (storage) MatType(R,C);
	mat = pyMap;

	memory->convertible = storage;
      }
    }
  };


}

#define IMLE_NO_TEMPLATES

#include "imle.hpp"

void setupEigenToNumpyConverters() {
    import_array();

    using namespace boost::python;

    to_python_converter<Eigen::MatrixXd,
    			        boopy::EigenMatrix_to_python_matrix<Eigen::MatrixXd> >();
    boopy::EigenMatrix_from_python_array<Eigen::MatrixXd>();

    to_python_converter<Eigen::VectorXd,
       			        boopy::EigenMatrix_to_python_matrix<Eigen::VectorXd> >();
    boopy::EigenMatrix_from_python_array<Eigen::VectorXd>();
}
