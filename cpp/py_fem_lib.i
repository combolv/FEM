%module fem_lib
%{
#include "./include/scene.h"
%}

%exception {
    try {
        $action
    } catch (const std::runtime_error& e) {
        PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
        SWIG_fail;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error.");
        SWIG_fail;
    }
}

%include <std_array.i>
%include <std_vector.i>
%include "./include/scene.h"
%include "./include/solver.h"

namespace std {
    %template(StdDoubleVector) vector<double>;
    %template(StdIntVector) vector<int>;
}
