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

from __future__ import print_function

header = """#
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
#"""

# Code generator for shared params (shared.py). Run under this folder with:
# python _shared_params_code_gen.py > shared.py


def _gen_param_header(paramTypeStr, name, doc, defaultValueStr, isValidFunctionStr):
    """
    Generates the header part for shared variables

    :param name: param name
    :param doc: param doc
    """
    template = '''class Has$Name(Params):
    """
    Mixin for param $name: $doc
    """

    $name = $paramType(Params._dummy(), "$name", "$doc", $isValid)

    def __init__(self):
        super(Has$Name, self).__init__()'''

    if defaultValueStr is not None:
        template += '''
        self._setDefault($name=$defaultValueStr)'''

    Name = name[0].upper() + name[1:]
    if isValidFunctionStr is None:
        isValidFunctionStr = str(None)
    return template \
        .replace("$paramType", paramTypeStr) \
        .replace("$name", name) \
        .replace("$Name", Name) \
        .replace("$doc", doc) \
        .replace("$defaultValueStr", str(defaultValueStr)) \
        .replace("$isValid", isValidFunctionStr)


def _gen_param_code(name, doc, defaultValueStr):
    """
    Generates Python code for a shared param class.

    :param name: param name
    :param doc: param doc
    :param defaultValueStr: string representation of the default value
    :return: code string
    """
    # TODO: How to correctly inherit instance attributes?
    template = '''
    def set$Name(self, value):
        """
        Sets the value of :py:attr:`$name`.
        """
        self._set($name=value)
        return self

    def get$Name(self):
        """
        Gets the value of $name or its default value.
        """
        return self.getOrDefault(self.$name)'''

    Name = name[0].upper() + name[1:]
    return template \
        .replace("$name", name) \
        .replace("$Name", Name) \
        .replace("$doc", doc) \
        .replace("$defaultValueStr", str(defaultValueStr))

if __name__ == "__main__":
    print(header)
    print("\n# DO NOT MODIFY THIS FILE! It was generated by _shared_params_code_gen.py.\n")
    print("from pyspark.ml.param import *\n\n")
    shared = [
        ("maxIter", "IntParam", "max number of iterations (>= 0).", None,
         "ParamValidators.gtEq(0)"),
        ("regParam", "FloatParam", "regularization parameter (>= 0).", None,
         "ParamValidators.gtEq(0)"),
        ("featuresCol", "StringParam", "features column name.", "'features'", None),
        ("labelCol", "StringParam", "label column name.", "'label'", None),
        ("predictionCol", "StringParam", "prediction column name.", "'prediction'", None),
        ("probabilityCol", "StringParam", "Column name for predicted class conditional " +
         "probabilities. Note: Not all models output well-calibrated probability estimates! These" +
         " probabilities should be treated as confidences, not precise probabilities.",
         "'probability'", None),
        ("rawPredictionCol", "StringParam", "raw prediction (a.k.a. confidence) column name.",
         "'rawPrediction'", None),
        ("inputCol", "StringParam", "input column name.", None, None),
        ("inputCols", "ListStringParam", "input column names.", None, None),
        ("outputCol", "StringParam", "output column name.", "self.uid + '__output'", None),
        ("numFeatures", "IntParam", "number of features.", None, "ParamValidators.gtEq(0)"),
        ("checkpointInterval", "IntParam", "set checkpoint interval (>= 1) or disable checkpoint" +
         " (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations.", None,
         "lambda interval: (interval == -1) or (interval >= 1)"),
        ("seed", "IntParam", "random seed.", "hash(type(self).__name__)", None),
        ("tol", "BooleanParam", "the convergence tolerance for iterative algorithms.", None, None),
        ("stepSize", "FloatParam", "Step size to be used for each iteration of optimization.", None,
         None),
        ("handleInvalid", "StringParam", "how to handle invalid entries. Options are skip (which " +
         "will filter out rows with bad values), or error (which will throw an errror). More " +
         "options may be added later.", None, "ParamValidators.inList(['skip', 'error'])"),
        ("elasticNetParam", "FloatParam", "the ElasticNet mixing parameter, in range [0, 1]. For " +
         "alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.", "0.0",
         "ParamValidators.inRange(0, 1)"),
        ("fitIntercept", "BooleanParam", "whether to fit an intercept term.", "True", None),
        ("standardization", "BooleanParam", "whether to standardize the training features before " +
         "fitting the model.", "True", None),
        ("thresholds", "ListFloatParam", "Thresholds in multi-class classification to adjust the " +
         "probability of predicting each class. Array must have length equal to the number of " +
         "classes, with values >= 0. The class with largest value p/t is predicted, where p is " +
         "the original probability of that class and t is the class' threshold.", None,
         "lambda lst: all(map(lambda t: t >= 0, lst))"),
        ("weightCol", "StringParam", "weight column name. If this is not set or empty, we treat " +
         "all instance weights as 1.0.", None, None),
        ("solver", "StringParam", "the solver algorithm for optimization. If this is not set or " +
         "empty, default value is 'auto'.", "'auto'", None)]

    code = []
    for name, paramClassStr, doc, defaultValueStr, isValidStr in shared:
        param_code = _gen_param_header(paramClassStr, name, doc, defaultValueStr, isValidStr)
        code.append(param_code + "\n" + _gen_param_code(name, doc, defaultValueStr))

    decisionTreeParams = [
        ("maxDepth", "IntParam", "Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf " +
         "node; depth 1 means 1 internal node + 2 leaf nodes.", "ParamValidators.gtEq(0)"),
        ("maxBins", "IntParam", "Max number of bins for" +
         " discretizing continuous features.  Must be >=2 and >= number of categories for any" +
         " categorical feature.", "ParamValidators.gtEq(2)"),
        ("minInstancesPerNode", "IntParam", "Minimum number of instances each child must have " +
         "after split. If a split causes the left or right child to have fewer than " +
         "minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.",
         "ParamValidators.gtEq(1)"),
        ("minInfoGain", "FloatParam", "Minimum information gain for a split to be considered at a "
         "tree node.", None),
        ("maxMemoryInMB", "IntParam", "Maximum memory in MB allocated to histogram aggregation.",
         "ParamValidators.gtEq(0)"),
        ("cacheNodeIds", "BooleanParam", "If false, the algorithm will pass trees to executors " +
         "to match instances with nodes. If true, the algorithm will cache node IDs for each " +
         "instance. Caching can speed up training of deeper trees. Users can set how often " +
         "should the cache be checkpointed or disable it by setting checkpointInterval.", None)]

    decisionTreeCode = '''class DecisionTreeParams(Params):
    """
    Mixin for Decision Tree parameters.
    """

    $dummyPlaceHolders

    def __init__(self):
        super(DecisionTreeParams, self).__init__()'''
    dtParamMethods = ""
    dummyPlaceholders = ""
    paramTemplate = """$name = $paramClass($owner, "$name", "$doc", $isValid)"""
    for name, paramClassStr, doc, isValidStr in decisionTreeParams:
        if isValidStr is None:
            isValidStr = str(None)
        variable = paramTemplate.replace("$name", name).replace("$doc", doc)\
            .replace("$paramClass", paramClassStr).replace("$isValid", isValidStr)
        dummyPlaceholders += variable.replace("$owner", "Params._dummy()") + "\n    "
        dtParamMethods += _gen_param_code(name, doc, None) + "\n"
    code.append(decisionTreeCode.replace("$dummyPlaceHolders", dummyPlaceholders) + "\n" +
                dtParamMethods)
    print("\n\n\n".join(code))
