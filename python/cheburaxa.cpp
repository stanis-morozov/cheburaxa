#include "../include/cheburaxa.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

template <class T, bool row_major>
std::tuple<py::array_t<T, row_major ? py::array::c_style : py::array::f_style>,
           py::array_t<T, row_major ? py::array::c_style : py::array::f_style>,
           py::array_t<std::size_t, row_major ? py::array::c_style : py::array::f_style>, 
           py::array_t<std::size_t, row_major ? py::array::c_style : py::array::f_style>,
           std::vector<cheburaxa::Info> >
           _approximate_matrix(py::buffer A_buf, std::size_t r)
{
    py::buffer_info A_info = A_buf.request();
    if (A_info.ndim != 2) throw std::runtime_error("A should be a matrix");
    std::size_t m = A_info.shape[0];
    std::size_t n = A_info.shape[1];
    const T * A = static_cast<const T*>(A_info.ptr);
    py::array_t<T, row_major ? py::array::c_style : py::array::f_style> pU({m, r});
    py::array_t<T, row_major ? py::array::c_style : py::array::f_style> pV({m, r});
    py::array_t<std::size_t, row_major ? py::array::c_style : py::array::f_style> pU_subsets({row_major ? m : (r + 1), row_major ? (r + 1) : m});
    py::array_t<std::size_t, row_major ? py::array::c_style : py::array::f_style> pV_subsets({row_major ? n : (r + 1), row_major ? (r + 1) : n});
    T *U = static_cast<T *>(pU.request().ptr);
    T *V = static_cast<T *>(pV.request().ptr);
    std::size_t *U_subsets = static_cast<std::size_t *>(pU_subsets.request().ptr);
    std::size_t *V_subsets = static_cast<std::size_t *>(pV_subsets.request().ptr);
    auto info = cheburaxa::approximate_matrix<row_major>(m, n, r, A, U, U_subsets, V, V_subsets);
    return std::make_tuple(pU, pV, pU_subsets, pV_subsets, info);
}

PYBIND11_MODULE(cheburaxa, m) {
    m.doc() = R"pbdoc(
        Cheburaxa: for Chebyshev low rank matrix approximation
        -----------------------

        .. currentmodule:: cheburaxa

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("approximate_matrix", [](py::array_t<float, py::array::c_style> A, std::size_t rank) { return _approximate_matrix<float, true>(A, rank); }, R"pbdoc(
        Perform Chebyshev low-rank approximation of matrix

        Approximate m x n matrix A by low-rank m x r and n x r factors  U and V, such that \|A - UV^T\|_C is minimized.
    )pbdoc");

    m.def("approximate_matrix", [](py::array_t<float, py::array::f_style> A, std::size_t rank) { return _approximate_matrix<float, false>(A, rank); }, R"pbdoc(
        Perform Chebyshev low-rank approximation of matrix

        Approximate m x n matrix A by low-rank m x r and n x r factors  U and V, such that \|A - UV^T\|_C is minimized.
    )pbdoc");

    m.def("approximate_matrix", [](py::array_t<double, py::array::f_style> A, std::size_t rank) { return _approximate_matrix<double, false>(A, rank); }, R"pbdoc(
        Perform Chebyshev low-rank approximation of matrix

        Approximate m x n matrix A by low-rank m x r and n x r factors  U and V, such that \|A - UV^T\|_C is minimized.
    )pbdoc");

    m.def("approximate_matrix", [](py::array_t<double, py::array::c_style | py::array::forcecast> A, std::size_t rank) { return _approximate_matrix<double, true>(A, rank); }, R"pbdoc(
        Perform Chebyshev low-rank approximation of matrix

        Approximate m x n matrix A by low-rank m x r and n x r factors  U and V, such that \|A - UV^T\|_C is minimized.
    )pbdoc");

    py::class_<cheburaxa::Info>(m, "Info")
    .def(py::init<>())
    .def_readwrite("err_info", &cheburaxa::Info::err_info)
    .def_readwrite("iter_num", &cheburaxa::Info::iter_num)
    .def_readwrite("approximation_error", &cheburaxa::Info::approximation_error);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#endif
}