#include "Genten_Pybind11_include.hpp"
#include "Genten_Pybind11_classes.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Driver.hpp"
#include "Kokkos_Core.hpp"

#include <pybind11/iostream.h>

namespace py = pybind11;
namespace Genten {
  using DTC = DistTensorContext<Genten::DefaultHostExecutionSpace>;
}

namespace {

template <typename ExecSpace, typename TensorType>
Genten::Ktensor
driver_impl(const TensorType& x,
            const Genten::Ktensor& u,
            Genten::AlgParams& algParams,
            const Genten::ptree& ptree,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::DistTensorContext<ExecSpace> dtc;
  auto xd = dtc.distributeTensor(x, algParams);
  Genten::KtensorT<ExecSpace> ud;
  const ttb_indx nc = u.ncomponents();
  const ttb_indx nd = u.ndims();
  if (nd > 0 && nc > 0) {
#ifdef HAVE_DIST
    ud = dtc.exportFromRoot(u);
#else
    // We do not create a mirror-view of u because we always want a copy,
    // since driver overwrites it and we don't want that in python.  And
    // dtc.exportFromRoot() in serial builds is just a mirror.
    ud = Genten::KtensorT<ExecSpace>(nc, nd);
    deep_copy(ud.weights(), u.weights());
    for (ttb_indx i=0; i<nd; ++i) {
      Genten::FacMatrixT<ExecSpace> A(u[i].nRows(), nc);
      deep_copy(A, u[i]);
      ud.set_factor(i, A);
    }
#endif
  }

  Genten::print_environment(xd, dtc, out);

  ud = Genten::driver(dtc, xd, ud, algParams, ptree, history, out);

  Genten::Ktensor ret =
    dtc.template importToAll<Genten::DefaultHostExecutionSpace>(ud);

  return ret;
}

template <typename TensorType>
Genten::Ktensor
driver_host(const TensorType& x,
            const Genten::Ktensor& u,
            Genten::AlgParams& algParams,
            const Genten::ptree& ptree,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = driver_impl<Genten::DefaultExecutionSpace>(x, u, algParams, ptree, history, out);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = driver_impl<Kokkos::Cuda>(x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = driver_impl<Kokkos::Experimental::HIP>(x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = driver_impl<Kokkos::Experimental::SYCL>(x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = driver_impl<Kokkos::OpenMP>(x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = driver_impl<Kokkos::Threads>(x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = driver_impl<Kokkos::Serial>(x, u, algParams, ptree, history, out);
#endif
  else
    Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));

  return ret;
}

template <typename ExecSpace, typename TensorType>
Genten::Ktensor
driver_impl(const Genten::DTC& dtc,
            const TensorType& x,
            const Genten::Ktensor& u,
            Genten::AlgParams& algParams,
            const Genten::ptree& ptree,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  auto xd = create_mirror_view(ExecSpace(), x);
  deep_copy(xd, x);
  Genten::KtensorT<ExecSpace> ud;
  const ttb_indx nc = u.ncomponents();
  const ttb_indx nd = u.ndims();
  if (nd > 0 && nc > 0) {
    // We do not create a mirror-view of u because we always want a copy,
    // since driver overwrites it and we don't want that in python.
    ud = Genten::KtensorT<ExecSpace>(nc, nd);
    deep_copy(ud.weights(), u.weights());
    for (ttb_indx i=0; i<nd; ++i) {
      Genten::FacMatrixT<ExecSpace> A(u[i].nRows(), nc);
      deep_copy(A, u[i]);
      ud.set_factor(i, A);
    }
  }

  Genten::DistTensorContext<ExecSpace> dtc2(dtc);

  Genten::print_environment(xd, dtc2, out);

  ud = Genten::driver(dtc2, xd, ud, algParams, ptree, history, out);

  Genten::Ktensor ret = create_mirror_view(ud);
  deep_copy(ret, ud);

  return ret;
}

template <typename TensorType>
Genten::Ktensor
driver_host(const Genten::DTC& dtc,
            const TensorType& x,
            const Genten::Ktensor& u,
            Genten::AlgParams& algParams,
            const Genten::ptree& ptree,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = driver_impl<Genten::DefaultExecutionSpace>(dtc, x, u, algParams, ptree, history, out);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = driver_impl<Kokkos::Cuda>(dtc, x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = driver_impl<Kokkos::Experimental::HIP>(dtc, x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = driver_impl<Kokkos::Experimental::SYCL>(dtc, x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = driver_impl<Kokkos::OpenMP>(dtc, x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = driver_impl<Kokkos::Threads>(dtc, x, u, algParams, ptree, history, out);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = driver_impl<Kokkos::Serial>(dtc, x, u, algParams, ptree, history, out);
#endif
  else
    Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));

  return ret;
}

}

PYBIND11_MODULE(_pygenten, m) {
    m.doc() = "_pygenten module";
    m.def("initializeKokkos", [](const int num_threads, const int num_devices, const int device_id) -> void {
        if(!Kokkos::is_initialized()) {
            Kokkos::InitializationSettings args;
            args.set_num_threads(num_threads);
            args.set_num_devices(num_devices);
            args.set_device_id(device_id);
            Kokkos::initialize(args);
        }
    }, "", pybind11::arg("num_threads") = -1, pybind11::arg("num_devices") = -1, pybind11::arg("device_id") = -1);
    m.def("finalizeKokkos", []() -> void {
        if(Kokkos::is_initialized())
            Kokkos::finalize();
    });

    m.def("initializeGenten", []() -> bool {
        return Genten::InitializeGenten();
    });
    m.def("finalizeGenten", []() -> void {
        Genten::FinalizeGenten();
    });

        m.def("driver", [](const Genten::Tensor& x,
                       const Genten::Ktensor& u0,
                       Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       py::scoped_ostream_redirect err_stream(
         std::cerr,                                // std::ostream&
         py::module_::import("sys").attr("stderr") // Python output
       );
       Genten::ptree ptree;
       Genten::PerfHistory perfInfo;
       Genten::Ktensor u = driver_host(x, u0, algParams, ptree, perfInfo, std::cout);
       return std::make_tuple(u, perfInfo);
    });
    m.def("driver", [](const Genten::Sptensor& x,
                       const Genten::Ktensor& u0,
                       Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       py::scoped_ostream_redirect err_stream(
         std::cerr,                                // std::ostream&
         py::module_::import("sys").attr("stderr") // Python output
       );
       Genten::ptree ptree;
       Genten::PerfHistory perfInfo;
       Genten::Ktensor u = driver_host(x, u0, algParams, ptree, perfInfo, std::cout);
       return std::make_tuple(u, perfInfo);
    });

    m.def("driver", [](const Genten::DTC& dtc,
                       const Genten::Tensor& x,
                       const Genten::Ktensor& u0,
                       Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       py::scoped_ostream_redirect err_stream(
         std::cerr,                                // std::ostream&
         py::module_::import("sys").attr("stderr") // Python output
       );
       Genten::ptree ptree;
       Genten::PerfHistory perfInfo;
       Genten::Ktensor u = driver_host(dtc, x, u0, algParams, ptree, perfInfo, std::cout);
       return std::make_tuple(u, perfInfo);
    });
    m.def("driver", [](const Genten::DTC& dtc,
                       const Genten::Sptensor& x,
                       const Genten::Ktensor& u0,
                       Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
       py::scoped_ostream_redirect stream(
         std::cout,                                // std::ostream&
         py::module_::import("sys").attr("stdout") // Python output
       );
       py::scoped_ostream_redirect err_stream(
         std::cerr,                                // std::ostream&
         py::module_::import("sys").attr("stderr") // Python output
       );
       Genten::ptree ptree;
       Genten::PerfHistory perfInfo;
       Genten::Ktensor u = driver_host(dtc, x, u0, algParams, ptree, perfInfo, std::cout);
       return std::make_tuple(u, perfInfo);
    });

    m.def("import_tensor", [](const std::string& fName) -> Genten::Tensor {
        Genten::Tensor X;
        Genten::import_tensor(fName, X);
        return X;
    });
    m.def("import_sptensor", [](const std::string& fName) -> Genten::Sptensor {
        Genten::Sptensor X;
        Genten::import_sptensor(fName, X);
        return X;
    });

    m.def("export_ktensor", [](const std::string& fName,
                               const Genten::Ktensor& u) -> void {
        Genten::export_ktensor(fName, u);
    });

    m.def("proc_rank", []() -> int {
        gt_assert(Genten::DistContext::initialized());
        return Genten::DistContext::rank();
    });
    m.def("num_procs", []() -> int {
        gt_assert(Genten::DistContext::initialized());
        return Genten::DistContext::nranks();
    });
    m.def("barrier", []() {
        gt_assert(Genten::DistContext::initialized());
        Genten::DistContext::Barrier();
    });

    pygenten_classes(m);
}
