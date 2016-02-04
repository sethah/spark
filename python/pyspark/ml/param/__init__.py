#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from abc import ABCMeta
import copy
import numpy as np
from pyspark.mllib.linalg import DenseVector, Vector

from pyspark import since
from pyspark.ml.util import Identifiable


__all__ = ['IntParam', 'FloatParam', 'StringParam', 'BooleanParam',
           'VectorParam', 'ListIntParam', 'ListFloatParam', 'ListStringParam',
           'Param', 'Params', 'ParamValidators']


class Param(object):
    """
    A param with self-contained documentation.

    .. versionadded:: 1.3.0
    """

    def __init__(self, parent, name, doc, isValid=None):
        if not isinstance(parent, Identifiable):
            raise TypeError("Parent must be an Identifiable but got type %s." % type(parent))
        self.parent = parent.uid
        self.name = str(name)
        self.doc = str(doc)
        self.isValid = isValid if isValid is not None else ParamValidators.alwaysTrue()

    def _copy_new_parent(self, parent):
        """Copy the current param to a new parent, must be a dummy param."""
        if self.parent == "undefined":
            param = copy.copy(self)
            param.parent = parent.uid
            return param
        else:
            raise ValueError("Cannot copy from non-dummy parent %s." % parent)

    def _validate(self, value):
        if not self.isValid(value):
            raise ValueError("{parent} parameter {name} given invalid value {value}"
                             .format(parent=self.parent, name=self.name, value=str(value)))

    def _convertAndValidate(self, value):
        self._validate(value)
        return value

    def __str__(self):
        return str(self.parent) + "__" + self.name

    def __repr__(self):
        return "Param(parent=%r, name=%r, doc=%r)" % (self.parent, self.name, self.doc)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, Param):
            return self.parent == other.parent and self.name == other.name
        else:
            return False


class IntParam(Param):
    """
    Specialized `Param` for integers.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        value = ParamValidators.primitiveConvert(value, int)
        self._validate(value)
        return value


class FloatParam(Param):
    """
    Specialized `Param` for floats.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        value = ParamValidators.primitiveConvert(value, float)
        self._validate(value)
        return value


class StringParam(Param):
    """
    Specialized `Param` for strings.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        value = ParamValidators.primitiveConvert(value, str)
        self._validate(value)
        return value


class BooleanParam(Param):
    """
    Specialized `Param` for Booleans.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        value = ParamValidators.primitiveConvert(value, bool)
        self._validate(value)
        return value


class ListIntParam(Param):
    """
    Specialized `Param` for lists of integers.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        if type(value) != list:
            value = ParamValidators.convertToList(value)

        if not all(map(lambda v: type(v) == int, value)):
            try:
                value = map(lambda v: int(v), value)
            except ValueError:
                raise TypeError("Could not convert %s to a list of integers" % value)
        self._validate(value)
        return value


class ListFloatParam(Param):
    """
    Specialized `Param` for lists of floats.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        if type(value) != list:
            value = ParamValidators.convertToList(value)

        if not all(map(lambda v: type(v) == float, value)):
            try:
                value = map(lambda v: float(v), value)
            except ValueError:
                raise TypeError("Could not convert %s to a list of floats" % value)
        self._validate(value)
        return value


class ListStringParam(Param):
    """
    Specialized `Param` for lists of strings.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        if type(value) != list:
            value = ParamValidators.convertToList(value)

        if not all(map(lambda v: type(v) == str, value)):
            try:
                value = map(lambda v: str(v), value)
            except ValueError:
                raise TypeError("Could not convert %s to a list of strings" % value)
        self._validate(value)
        return value


class VectorParam(Param):
    """
    Specialized `Param` for Vector types.

    .. versionadded:: 2.0.0
    """

    def _convertAndValidate(self, value):
        if not isinstance(value, Vector):
            try:
                value = DenseVector(value)
            except:
                raise TypeError("Could not convert %s to a Vector" % value)
        self._validate(value)
        return value


class ParamValidators(object):

    @staticmethod
    def alwaysTrue():
        return lambda value: True

    @staticmethod
    def primitiveConvert(value, primitiveType):
        if type(value) != primitiveType:
            try:
                value = primitiveType(value)
            except ValueError:
                raise TypeError("Could not convert %s to a %s" % (value, primitiveType.__name__))
        return value

    @staticmethod
    def convertToList(value):
        if type(value) == np.ndarray:
            return list(value)
        elif isinstance(value, Vector):
            return value.toArray()
        else:
            raise TypeError("Could not convert %s to list" % value)

    @staticmethod
    def gt(lowerBound):
        return lambda value: value > lowerBound

    @staticmethod
    def gtEq(lowerBound):
        return lambda value: value >= lowerBound

    @staticmethod
    def lt(lowerBound):
        return lambda value: value < lowerBound

    @staticmethod
    def ltEq(lowerBound):
        return lambda value: value <= lowerBound

    @staticmethod
    def inRange(lowerBound, upperBound, lowerInclusive=True, upperInclusive=True):
        def inRangeFunction(x):
            lowerValid = (x >= lowerBound) if lowerInclusive else (x > lowerBound)
            upperValid = (x <= upperBound) if upperInclusive else (x < upperBound)
            return lowerValid and upperValid
        return inRangeFunction

    @staticmethod
    def inList(allowed):
        return lambda value: value in allowed

    @staticmethod
    def listLengthGt(lowerBound):
        return lambda lst: len(lst) > lowerBound


class Params(Identifiable):
    """
    Components that take parameters. This also provides an internal
    param map to store parameter values attached to the instance.

    .. versionadded:: 1.3.0
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        super(Params, self).__init__()
        #: internal param map for user-supplied values param map
        self._paramMap = {}

        #: internal param map for default values
        self._defaultParamMap = {}

        #: value returned by :py:func:`params`
        self._params = None

        # Copy the params from the class to the object
        self._copy_params()

    def _copy_params(self):
        """
        Copy all params defined on the class to current object.
        """
        cls = type(self)
        src_name_attrs = [(x, getattr(cls, x)) for x in dir(cls)]
        src_params = list(filter(lambda nameAttr: isinstance(nameAttr[1], Param), src_name_attrs))
        for name, param in src_params:
            setattr(self, name, param._copy_new_parent(self))

    @property
    @since("1.3.0")
    def params(self):
        """
        Returns all params ordered by name. The default implementation
        uses :py:func:`dir` to get all attributes of type
        :py:class:`Param`.
        """
        if self._params is None:
            self._params = list(filter(lambda attr: isinstance(attr, Param),
                                       [getattr(self, x) for x in dir(self) if x != "params"]))
        return self._params

    @since("1.4.0")
    def explainParam(self, param):
        """
        Explains a single param and returns its name, doc, and optional
        default value and user-supplied value in a string.
        """
        param = self._resolveParam(param)
        values = []
        if self.isDefined(param):
            if param in self._defaultParamMap:
                values.append("default: %s" % self._defaultParamMap[param])
            if param in self._paramMap:
                values.append("current: %s" % self._paramMap[param])
        else:
            values.append("undefined")
        valueStr = "(" + ", ".join(values) + ")"
        return "%s: %s %s" % (param.name, param.doc, valueStr)

    @since("1.4.0")
    def explainParams(self):
        """
        Returns the documentation of all params with their optionally
        default values and user-supplied values.
        """
        return "\n".join([self.explainParam(param) for param in self.params])

    @since("1.4.0")
    def getParam(self, paramName):
        """
        Gets a param by its name.
        """
        param = getattr(self, paramName)
        if isinstance(param, Param):
            return param
        else:
            raise ValueError("Cannot find param with name %s." % paramName)

    @since("1.4.0")
    def isSet(self, param):
        """
        Checks whether a param is explicitly set by user.
        """
        param = self._resolveParam(param)
        return param in self._paramMap

    @since("1.4.0")
    def hasDefault(self, param):
        """
        Checks whether a param has a default value.
        """
        param = self._resolveParam(param)
        return param in self._defaultParamMap

    @since("1.4.0")
    def isDefined(self, param):
        """
        Checks whether a param is explicitly set by user or has
        a default value.
        """
        return self.isSet(param) or self.hasDefault(param)

    @since("1.4.0")
    def hasParam(self, paramName):
        """
        Tests whether this instance contains a param with a given
        (string) name.
        """
        param = self._resolveParam(paramName)
        return param in self.params

    @since("1.4.0")
    def getOrDefault(self, param):
        """
        Gets the value of a param in the user-supplied param map or its
        default value. Raises an error if neither is set.
        """
        param = self._resolveParam(param)
        if param in self._paramMap:
            return self._paramMap[param]
        else:
            return self._defaultParamMap[param]

    @since("1.4.0")
    def extractParamMap(self, extra=None):
        """
        Extracts the embedded default param values and user-supplied
        values, and then merges them with extra values from input into
        a flat param map, where the latter value is used if there exist
        conflicts, i.e., with ordering: default param values <
        user-supplied values < extra.

        :param extra: extra param values
        :return: merged param map
        """
        if extra is None:
            extra = dict()
        paramMap = self._defaultParamMap.copy()
        paramMap.update(self._paramMap)
        paramMap.update(extra)
        return paramMap

    @since("1.4.0")
    def copy(self, extra=None):
        """
        Creates a copy of this instance with the same uid and some
        extra params. The default implementation creates a
        shallow copy using :py:func:`copy.copy`, and then copies the
        embedded and extra parameters over and returns the copy.
        Subclasses should override this method if the default approach
        is not sufficient.

        :param extra: Extra parameters to copy to the new instance
        :return: Copy of this instance
        """
        if extra is None:
            extra = dict()
        that = copy.copy(self)
        that._paramMap = self.extractParamMap(extra)
        return that

    def _shouldOwn(self, param):
        """
        Validates that the input param belongs to this Params instance.
        """
        if not (self.uid == param.parent and self.hasParam(param.name)):
            raise ValueError("Param %r does not belong to %r." % (param, self))

    def _resolveParam(self, param):
        """
        Resolves a param and validates the ownership.

        :param param: param name or the param instance, which must
                      belong to this Params instance
        :return: resolved param instance
        """
        if isinstance(param, Param):
            self._shouldOwn(param)
            return param
        elif isinstance(param, str):
            return self.getParam(param)
        else:
            raise ValueError("Cannot resolve %r as a param." % param)

    @staticmethod
    def _dummy():
        """
        Returns a dummy Params instance used as a placeholder to
        generate docs.
        """
        dummy = Params()
        dummy.uid = "undefined"
        return dummy

    def _set(self, **kwargs):
        """
        Sets user-supplied params.
        """
        for param, value in kwargs.items():
            p = getattr(self, param)
            value = p._convertAndValidate(value)
            self._paramMap[getattr(self, param)] = value

        return self

    def _setDefault(self, **kwargs):
        """
        Sets default params.
        """
        for param, value in kwargs.items():
            self._defaultParamMap[getattr(self, param)] = value
        return self

    def _copyValues(self, to, extra=None):
        """
        Copies param values from this instance to another instance for
        params shared by them.

        :param to: the target instance
        :param extra: extra params to be copied
        :return: the target instance with param values copied
        """
        if extra is None:
            extra = dict()
        paramMap = self.extractParamMap(extra)
        for p in self.params:
            if p in paramMap and to.hasParam(p.name):
                to._set(**{p.name: paramMap[p]})
        return to
