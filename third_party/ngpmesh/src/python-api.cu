#include<NGPMesh.cuh>
#include<pybind11/pybind11.h>
#include<pybind11/chrono.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>
#include<pybind11/eigen.h>
#include<pybind11/functional.h>

namespace py = pybind11;

PYBIND11_MODULE(pyngpmesh, m){
    py::enum_<SDFCalcMode>(m, "SDFCalcMode")
        .value("WaterTight", SDFCalcMode::WATERTIGHT)
        .value("RayStab", SDFCalcMode::RAYSTAB)
        .value("PathEscape", SDFCalcMode::PATHESCAPE);

    py::class_<Triangle>(m, "Triangle")
        .def_readwrite("a", &Triangle::a)
        .def_readwrite("b", &Triangle::b)
        .def_readwrite("c", &Triangle::c);

    py::class_<BoundingBox>(m, "BoundingBox")
        .def(py::init<>())
        .def_readwrite("min", &BoundingBox::min)
        .def_readwrite("max", &BoundingBox::max);

    py::class_<NGPMesh>(m, "NGPMesh")
        .def(py::init<>())
        .def(py::init<std::vector<Eigen::Matrix3f>&>())
        .def(py::init<std::vector<Eigen::Vector3f>&, std::vector<Eigen::Vector<int, 3>>&>())
        .def("load_mesh", py::overload_cast<std::vector<Eigen::Matrix3f>&>(&NGPMesh::load_mesh))
        .def("load_mesh", py::overload_cast<std::vector<Eigen::Vector3f>&, std::vector<Eigen::Vector<int, 3>>&>(&NGPMesh::load_mesh))
        .def("unsigned_distance", &NGPMesh::unsigned_distance)
        .def("signed_distance", &NGPMesh::signed_distance, "calculate sdf value using ngp method",
            py::arg("positions"),
            py::arg("mode") = SDFCalcMode::RAYSTAB)
        .def("nearest_point", &NGPMesh::nearest_point)
        .def("trace", &NGPMesh::trace);
    
}