import _imle

from numpy import array


class ImleParam(_imle.ImleParam):
    fields = [f for f in dir(_imle.ImleParam) if not f.startswith('_')]

    def __init__(self, d, D, **kwargs):
        _imle.ImleParam.__init__(self, d, D)

        for field, value in kwargs.items():
            setattr(self, field, value)

    @property
    def Psi0(self):
        return _imle.ImleParam._get_psi0(self)

    @Psi0.setter
    def Psi0(self, value):
        _imle.ImleParam._set_psi0(self, array(value))


    def __repr__(self):
        return repr({f: getattr(self, f) for f in ImleParam.fields})
