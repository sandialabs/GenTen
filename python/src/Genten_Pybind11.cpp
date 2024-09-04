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

  ud = Genten::driver(dtc, xd, ud, algParams, history, out);

  Genten::Ktensor ret =
    dtc.template importToAll<Genten::DefaultHostExecutionSpace>(ud);

  return ret;
}

template <typename TensorType>
Genten::Ktensor
driver_host(const TensorType& x,
            const Genten::Ktensor& u,
            Genten::AlgParams& algParams,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = driver_impl<Genten::DefaultExecutionSpace>(x, u, algParams, history, out);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = driver_impl<Kokkos::Cuda>(x, u, algParams, history, out);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = driver_impl<Kokkos::Experimental::HIP>(x, u, algParams, history, out);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = driver_impl<Kokkos::Experimental::SYCL>(x, u, algParams, history, out);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = driver_impl<Kokkos::OpenMP>(x, u, algParams, history, out);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = driver_impl<Kokkos::Threads>(x, u, algParams, history, out);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = driver_impl<Kokkos::Serial>(x, u, algParams, history, out);
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

  ud = Genten::driver(dtc2, xd, ud, algParams, history, out);

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
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = driver_impl<Genten::DefaultExecutionSpace>(dtc, x, u, algParams, history, out);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = driver_impl<Kokkos::Cuda>(dtc, x, u, algParams, history, out);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = driver_impl<Kokkos::Experimental::HIP>(dtc, x, u, algParams, history, out);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = driver_impl<Kokkos::Experimental::SYCL>(dtc, x, u, algParams, history, out);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = driver_impl<Kokkos::OpenMP>(dtc, x, u, algParams, history, out);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = driver_impl<Kokkos::Threads>(dtc, x, u, algParams, history, out);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = driver_impl<Kokkos::Serial>(dtc, x, u, algParams, history, out);
#endif
  else
    Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));

  return ret;
}

}

PYBIND11_MODULE(_pygenten, m) {
  m.doc() = R"(
     Module providing Python wrappers for GenTen.

     GenTen is a tool providing Canonical Polyadic (CP) tensor decomposition
     capabilities for sparse and dense tensors.  It provides data structures
     for storing sparse and dense tensors, and several solver methods
     for computing CP decompositions of those tensors.  It leverages Kokkos
     for shared-memory parallelism on CPU and GPU architectures, and MPI for
     distributed memory parallelism.)";
  m.def("initializeKokkos", [](const int num_threads, const int device_id) -> void {
      if(!Kokkos::is_initialized()) {
        Kokkos::InitializationSettings args;
        args.set_num_threads(num_threads);
        args.set_device_id(device_id);
        Kokkos::initialize(args);
      }
    }, R"(
    Initialize Kokkos to set up the shared memory parallel environment.

    Users should not generally call this, but instead call 'initializeGenten'.)", pybind11::arg("num_threads") = -1, pybind11::arg("device_id") = -1);
  m.def("finalizeKokkos", []() -> void {
      if(Kokkos::is_initialized())
        Kokkos::finalize();
    }, R"(
    Finalize Kokkos to tear down the shared memory parallel environment.

    Users should not generally call this, but instead call 'finalizeGenten'.)");

  m.def("initializeGenten", []() -> bool {
      return Genten::InitializeGenten();
    }, R"(
    Initialize GenTen to set up the parallel environment.

    Returns True if this call resulted in the initialization and False if it
    was already initialized.)");
  m.def("finalizeGenten", []() -> void {
      Genten::FinalizeGenten();
    }, R"(
    Finalize GenTen to tear down the parallel environment.

    All GenTen data structures must be destroyed before calling this.)");

  m.def("driver", [](const Genten::Tensor& x, const Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
      py::scoped_ostream_redirect stream(
        std::cout,                                // std::ostream&
        py::module_::import("sys").attr("stdout") // Python output
        );
      py::scoped_ostream_redirect err_stream(
        std::cerr,                                // std::ostream&
        py::module_::import("sys").attr("stderr") // Python output
       );
      Genten::PerfHistory perfInfo;
      Genten::Ktensor u = driver_host(x, u0, algParams, perfInfo, std::cout);
      return std::make_tuple(u, perfInfo);
    }, R"(
    Low-level driver for calling GenTen's solver methods on dense tensors.

    Users should usually not call this directly, but rather the provided
    solver wrappers for individual methods.

    Parameters:
      * X: the tensor to compute the CP decomposition from.
      * u0: the initial guess, which may be empty, in which case GenTen will
        compute a random initial guess.
      * algParams: algorithmic parameters controlling which algorithm is chosen
        as well as its solver parameters.

    Returns a tuple of the Ktensor solution and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));
    m.def("driver", [](const Genten::Sptensor& x, const Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
        py::scoped_ostream_redirect stream(
          std::cout,                                // std::ostream&
          py::module_::import("sys").attr("stdout") // Python output
          );
        py::scoped_ostream_redirect err_stream(
          std::cerr,                                // std::ostream&
          py::module_::import("sys").attr("stderr") // Python output
          );
        Genten::PerfHistory perfInfo;
        Genten::Ktensor u = driver_host(x, u0, algParams, perfInfo, std::cout);
        return std::make_tuple(u, perfInfo);
      }, R"(
    Low-level driver for calling GenTen's solver methods on sparse tensors.

    Users should usually not call this directly, but rather the provided
    solver wrappers for individual methods.

    Parameters:
      * X: the tensor to compute the CP decomposition from.
      * u0: the initial guess, which may be empty, in which case GenTen will
        compute a random initial guess.
      * algParams: algorithmic parameters controlling which algorithm is chosen
        as well as its solver parameters.

    Returns a tuple of the Ktensor solution and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));

    m.def("driver", [](const Genten::DTC& dtc, const Genten::Tensor& x, const Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
        py::scoped_ostream_redirect stream(
          std::cout,                                // std::ostream&
          py::module_::import("sys").attr("stdout") // Python output
          );
        py::scoped_ostream_redirect err_stream(
          std::cerr,                                // std::ostream&
          py::module_::import("sys").attr("stderr") // Python output
          );
        Genten::PerfHistory perfInfo;
        Genten::Ktensor u = driver_host(dtc, x, u0, algParams, perfInfo, std::cout);
        return std::make_tuple(u, perfInfo);
      }, R"(
    Low-level driver for calling GenTen's solver methods on dense tensors.

    Users should usually not call this directly, but rather the provided
    solver wrappers for individual methods.

    Parameters:
      * dtc: distributed tensor context containing informatio on how the
        the tensor was distributed in parallel.
      * X: the tensor to compute the CP decomposition from.
      * u0: the initial guess, which may be empty, in which case GenTen will
        compute a random initial guess.
      * algParams: algorithmic parameters controlling which algorithm is chosen
        as well as its solver parameters.

    Returns a tuple of the Ktensor solution and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("dtc"), pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));
    m.def("driver", [](const Genten::DTC& dtc, const Genten::Sptensor& x, const Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
        py::scoped_ostream_redirect stream(
          std::cout,                                // std::ostream&
          py::module_::import("sys").attr("stdout") // Python output
          );
        py::scoped_ostream_redirect err_stream(
          std::cerr,                                // std::ostream&
          py::module_::import("sys").attr("stderr") // Python output
          );
        Genten::PerfHistory perfInfo;
        Genten::Ktensor u = driver_host(dtc, x, u0, algParams, perfInfo, std::cout);
        return std::make_tuple(u, perfInfo);
    }, R"(
    Low-level driver for calling GenTen's solver methods on sparsetensors.

    Users should usually not call this directly, but rather the provided
    solver wrappers for individual methods.

    Parameters:
      * dtc: distributed tensor context containing informatio on how the
        the tensor was distributed in parallel.
      * X: the tensor to compute the CP decomposition from.
      * u0: the initial guess, which may be empty, in which case GenTen will
        compute a random initial guess.
      * algParams: algorithmic parameters controlling which algorithm is chosen
        as well as its solver parameters.

    Returns a tuple of the Ktensor solution and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("dtc"), pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));

    m.def("import_ktensor", [](const std::string& fName) -> Genten::Ktensor {
        Genten::Ktensor u;
        Genten::import_ktensor(fName, u);
        return u;
    }, R"(
    Read and return a Kruskal tensor from a given file.)", pybind11::arg("file"));
    m.def("import_tensor", [](const std::string& fName) -> Genten::Tensor {
        Genten::Tensor X;
        Genten::import_tensor(fName, X);
        return X;
    }, R"(
    Read and return a dense tensor from a given file.)", pybind11::arg("file"));
    m.def("import_sptensor", [](const std::string& fName) -> Genten::Sptensor {
        Genten::Sptensor X;
        Genten::import_sptensor(fName, X);
        return X;
    }, R"(
    Read and return a sparse tensor from a given file.)", pybind11::arg("file"));

    m.def("export_ktensor", [](const std::string& fName, const Genten::Ktensor& u) -> void {
        Genten::export_ktensor(fName, u);
    }, R"(
    Write a given Ktensor to the given file.)", pybind11::arg("file"), pybind11::arg("u"));
    m.def("export_tensor", [](const std::string& fName, const Genten::Tensor& X) -> void {
        Genten::export_tensor(fName, X);
    }, R"(
    Write a given Tensor to the given file.)", pybind11::arg("file"), pybind11::arg("X"));
    m.def("export_sptensor", [](const std::string& fName, const Genten::Sptensor& X) -> void {
        Genten::export_sptensor(fName, X);
    }, R"(
    Write a given Tensor to the given file.)", pybind11::arg("file"), pybind11::arg("X"));

    m.def("proc_rank", []() -> int {
        gt_assert(Genten::DistContext::initialized());
        return Genten::DistContext::rank();
    }, R"(
    Return MPI rank of the this processor.)");
    m.def("num_procs", []() -> int {
        gt_assert(Genten::DistContext::initialized());
        return Genten::DistContext::nranks();
    }, R"(
    Return total number of MPI ranks.)");
    m.def("barrier", []() {
        gt_assert(Genten::DistContext::initialized());
        Genten::DistContext::Barrier();
    }, R"(
    MPI barrier to synchronize MPI ranks.)");

    pygenten_classes(m);
}
