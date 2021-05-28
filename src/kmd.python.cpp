#include "kmd.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(kmd, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<ParticleVector>(m, "ParticleVector");
    m.def("init_kmd", &init_kmd);
    m.def("finalize_kmd", &finalize_kmd);
    m.def("init_particles", &init_particles);
    m.def("interact", &interact);
}
