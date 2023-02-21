#include "Genten_Pybind11_include.hpp"
#include "Genten_Pybind11_classes.hpp"
#include "Kokkos_Core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pygenten, m) {
    m.doc() = "_pygenten module";
    m.def("initializeKokkos", [](const int num_threads, const int num_devices, const int device_id) -> void {
        if(!Kokkos::is_initialized()) {
            Kokkos::InitArguments args(num_threads,num_devices,device_id);
            Kokkos::initialize(args);
        }
    }, "", pybind11::arg("num_threads") = -1, pybind11::arg("num_devices") = -1, pybind11::arg("device_id") = -1);
    m.def("finalizeKokkos", []() -> void {
        if(Kokkos::is_initialized())
            Kokkos::finalize();
    });
    pygenten_classes(m);
}
