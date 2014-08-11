from numpy import array, ndarray

import _imle

from .param import ImleParam


class Imle(object):
    """ Imle Object Learning f: z -> x (with |z| = d and |x| = D). """

    def __init__(self, d, D, **kwargs):
        self._param = ImleParam(d, D, **kwargs)
        self._imle_obj = _imle.Imle(d, D, self._param)

    def reset(self):
        """ Resets the learning. """
        self._imle_obj.reset()

    def update(self, z, x):
        """ Update the IMLE model with one (or more) training sample(s).

        :param numpy.array z: z must be shaped as (d,) or (N, d)
        :param numpy.array x: x must be shaped as (D,) or (N, D)

        """
        x = self._reshape_x(x)
        z = self._reshape_z(z)

        if x.shape[0] != z.shape[0]:
            msg = ('z and x must have the number of rows! '
                   '(z.shape: {}, x.shape: {})').format(z.shape, x.shape)
            raise ValueError(msg)

        for xx, zz in zip(x, z):
            self._imle_obj.update(zz, xx)

    def predict(self, z, strongest=False,
                multiple=False,
                var=False, weight=False,
                jacobian=False,
                error_reduction=False,
                error_reduction_derivative=False):
        """ Forward prediction (z -> x).

        :param numpy.array z: z must be shaped as (d,) or (N, d)
        :param bool strongest: returns the strongest prediction
        :param bool multiple: returns multiple predictions
        :param bool var, weight, jacobian, error_reduction, error_reduction_derivative: returns the associated value in addition to the prediction (when set returns a dictionnary).

        """
        return self._predict(z, inverse=False,
                             multiple=multiple,
                             strongest=strongest,
                             var=var, weight=weight,
                             jacobian=jacobian,
                             error_reduction=error_reduction,
                             error_reduction_derivative=error_reduction_derivative)

    def predict_inverse(self, x, var=False, weight=False):
        """ Inverse prediction (x -> z).

        :param numpy.array x: z must be shaped as (D,) or (N, D)
        :param bool var, weight: returns the associated value in addition to the prediction (when set returns a dictionnary).

        """
        return self._predict(x, inverse=True,
                             var=var, weight=weight)

    @property
    def number_of_experts(self):
        """ Returns the number of expert created. """
        return self._imle_obj.getNumberOfExperts()

    @property
    def param(self):
        """ Returns a dictionnary of all IMLE parameters. """
        return self._param

    @property
    def d(self):
        """ Input dimensionality. """
        return self._imle_obj.inputDim()

    @property
    def D(self):
        """ Output dimensionality. """
        return self._imle_obj.outputDim()

    def _predict(self, din, inverse=False, **kwargs):
        multiple = False if 'multiple' not in kwargs else kwargs['multiple']

        din = array(din)
        use_vector = len(din.shape) == 1
        din = self._reshape_x(din) if inverse else self._reshape_z(din)

        func = (self._inverse_predict if inverse else
                (self._multiple_forward_predict if multiple else
                 self._forward_predict))

        d = [func(d, **kwargs) for d in din]
        data = {}
        for k, v in d[0].items():
            data[k] = array([dd[k] for dd in d])

        if use_vector:
            for k, v in data.items():
                data[k] = v[0]

        if data.keys() == ['prediction']:
            data = data['prediction']

        return data

    def _forward_predict(self, z, strongest=False,
                         var=False, weight=False,
                         jacobian=False,
                         error_reduction=False,
                         error_reduction_derivative=False,
                         **kwargs):
        imle = self._imle_obj
        data = {}
        data['prediction'] = (imle.predictStrongest(z).flatten() if strongest
                              else imle.predict(z).flatten())
        if var:
            data['var'] = imle.getPredictionVar().flatten()
        if weight:
            data['weight'] = imle.getPredictionWeight().flatten()
        if jacobian:
            data['jacobian'] = imle.getPredictionJacobian()
        if error_reduction:
            data['error_reduction'] = imle.getPredictionErrorReduction().flatten()
        if error_reduction_derivative:
            data['error_reduction_derivative'] = imle.getPredictionErrorReductionDerivative()

        return data

    def _multiple_forward_predict(self, z,
                                  var=False, weight=False,
                                  jacobian=False,
                                  error_reduction=False,
                                  error_reduction_derivative=False,
                                  **kwargs):
        imle = self._imle_obj
        imle.predictMultiple(z)

        data = {}
        data['prediction'] = imle.getMultiplePredictions()
        if var:
            data['var'] = imle.getMultiplePredictionsVar()
        if weight:
            data['weight'] = imle.getMultiplePredictionsWeight()
        if jacobian:
            data['jacobian'] = imle.getMultiplePredictionsJacobian()
        if error_reduction:
            data['error_reduction'] = imle.getMultiplePredictionErrorReduction()
        if error_reduction_derivative:
            data['error_reduction_derivative'] = imle.getMultiplePredictionErrorReductionDerivative()

        return data

    def _inverse_predict(self, x, var=False, weight=False, **kwargs):
        imle = self._imle_obj
        imle.predictInverse(x)

        data = {}
        data['prediction'] = array(imle.getInversePredictions())
        if var:
            data['var'] = array(imle.getInversePredictionsVar())
        if weight:
            data['weight'] = array(imle.getInversePredictionsWeight())
        return data

    def _reshape(self, a, size):
        if not isinstance(a, ndarray):
            if not isinstance(a, list):
                raise ValueError('Arguments must be arrays!')

            a = array(a)

        if not (len(a.shape) in (1, 2) and a.shape[-1] == size):
            msg = 'Array must be shaped as ({}, ) or (N, {})'.format(size, size)
            raise ValueError(msg)

        return a.reshape(-1, size)

    def _reshape_z(self, z):
        return self._reshape(z, self.d)

    def _reshape_x(self, x):
        return self._reshape(x, self.D)
