#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "types.h"
#include "lsq.h"

namespace py=pybind11;

// el primer argumento es el nombre...
PYBIND11_MODULE(metnum, m) {
    py::class_<LeastSquareMethod>(m, "LeastSquareMethod")
        .def(py::init<>())
        .def("ajustar", &LeastSquareMethod::ajustar);

}
