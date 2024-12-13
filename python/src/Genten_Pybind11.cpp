#include "Genten_Pybind11_include.hpp"
#include "Genten_Pybind11_classes.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Driver.hpp"
#include "Genten_Online_GCP.hpp"
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
            Genten::Ktensor& u0,
            Genten::AlgParams& algParams,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::DistTensorContext<ExecSpace> dtc;
  auto xd = dtc.distributeTensor(x, algParams);
  auto u0d = dtc.exportFromRoot(u0);

  Genten::print_environment(xd, dtc, out);

  Genten::KtensorT<ExecSpace> ud = Genten::driver(dtc, xd, u0d, algParams, history, out);

  Genten::Ktensor ret =
    dtc.template importToAll<Genten::DefaultHostExecutionSpace>(ud);
  u0 = dtc.template importToAll<Genten::DefaultHostExecutionSpace>(u0d);

  return ret;
}

template <typename TensorType>
Genten::Ktensor
driver_host(const TensorType& x,
            Genten::Ktensor& u0,
            Genten::AlgParams& algParams,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = driver_impl<Genten::DefaultExecutionSpace>(x, u0, algParams, history, out);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = driver_impl<Kokkos::Cuda>(x, u0, algParams, history, out);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = driver_impl<Kokkos::Experimental::HIP>(x, u0, algParams, history, out);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = driver_impl<Kokkos::Experimental::SYCL>(x, u0, algParams, history, out);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = driver_impl<Kokkos::OpenMP>(x, u0, algParams, history, out);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = driver_impl<Kokkos::Threads>(x, u0, algParams, history, out);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = driver_impl<Kokkos::Serial>(x, u0, algParams, history, out);
#endif
  else
    Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));

  return ret;
}

template <typename ExecSpace, typename TensorType>
Genten::Ktensor
driver_impl(const Genten::DTC& dtc,
            const TensorType& x,
            Genten::Ktensor& u0,
            Genten::AlgParams& algParams,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  auto xd = create_mirror_view(ExecSpace(), x);
  auto u0d = create_mirror_view(ExecSpace(), u0);
  deep_copy(xd, x);
  deep_copy(u0d, u0);

  Genten::DistTensorContext<ExecSpace> dtc2(dtc);

  Genten::print_environment(xd, dtc2, out);

  Genten::KtensorT<ExecSpace> ud = Genten::driver(dtc2, xd, u0d, algParams, history, out);

  Genten::Ktensor ret = create_mirror_view(ud);
  u0 = create_mirror_view(u0d);
  deep_copy(ret, ud);
  deep_copy(u0, u0d);

  return ret;
}

template <typename TensorType>
Genten::Ktensor
driver_host(const Genten::DTC& dtc,
            const TensorType& x,
            Genten::Ktensor& u0,
            Genten::AlgParams& algParams,
            Genten::PerfHistory& history,
            std::ostream& out)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = driver_impl<Genten::DefaultExecutionSpace>(dtc, x, u0, algParams, history, out);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = driver_impl<Kokkos::Cuda>(dtc, x, u0, algParams, history, out);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = driver_impl<Kokkos::Experimental::HIP>(dtc, x, u0, algParams, history, out);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = driver_impl<Kokkos::Experimental::SYCL>(dtc, x, u0, algParams, history, out);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = driver_impl<Kokkos::OpenMP>(dtc, x, u0, algParams, history, out);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = driver_impl<Kokkos::Threads>(dtc, x, u0, algParams, history, out);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = driver_impl<Kokkos::Serial>(dtc, x, u0, algParams, history, out);
#endif
  else
    Genten::error("Invalid execution space: " + std::string(Genten::Execution_Space::names[algParams.exec_space]));

  return ret;
}

template <typename ExecSpace, typename TensorType>
Genten::Ktensor
online_gcp_impl(const std::vector<TensorType>& X, 
                const TensorType& X0, 
                const Genten::Ktensor& u,
                Genten::AlgParams& algParams,
                Genten::AlgParams& temporalAlgParams, 
                Genten::AlgParams& spatialAlgParams,
                std::ostream& out,
                Genten::Array& fest,
                Genten::Array& ften)
{
  auto X0d = create_mirror_view(ExecSpace(), X0);
  deep_copy(X0d, X0);
  ttb_indx n = X.size();
  std::vector<decltype(X0d)> Xd(n);
  for (ttb_indx i=0; i<n; ++i) {
    Xd[i] = create_mirror_view(ExecSpace(), X[i]);
    deep_copy(Xd[i], X[i]);
  }
  Genten::KtensorT<ExecSpace> ud;
  const ttb_indx nc = u.ncomponents();
  const ttb_indx nd = u.ndims();
  if (nd > 0 && nc > 0) {
    // We do not create a mirror-view of u because we always want a copy,
    // since online_gcp overwrites it and we don't want that in python.
    ud = Genten::KtensorT<ExecSpace>(nc, nd);
    deep_copy(ud.weights(), u.weights());
    for (ttb_indx i=0; i<nd; ++i) {
      Genten::FacMatrixT<ExecSpace> A(u[i].nRows(), nc);
      deep_copy(A, u[i]);
      ud.set_factor(i, A);
    }
  }

  // Fixup algParams
  const bool sparse = std::is_same_v<TensorType,Genten::SptensorT<ExecSpace> > ||
    algParams.streaming_solver == Genten::GCP_Streaming_Solver::SGD;
  algParams.method = Genten::Solver_Method::Online_GCP;
  algParams.sparse = sparse;
  algParams.fixup<ExecSpace>(out);

  // Fixup temporalAlgParams
  temporalAlgParams.sparse = sparse;
  temporalAlgParams.loss_function_type = algParams.loss_function_type;
  temporalAlgParams.fixup<ExecSpace>(out);

  // Fixup spatialAlgParams
  spatialAlgParams.sparse = sparse;
  spatialAlgParams.loss_function_type = algParams.loss_function_type;
  spatialAlgParams.fixup<ExecSpace>(out);

  Genten::Array fest_d, ften_d;
  Genten::online_gcp(Xd, X0d, ud, algParams, temporalAlgParams, spatialAlgParams, std::cout, fest_d, ften_d);
  Genten::Ktensor ret = create_mirror_view(ud);
  fest = create_mirror_view(fest_d);
  ften = create_mirror_view(ften_d);
  deep_copy(ret, ud);
  deep_copy(fest, fest_d);
  deep_copy(ften, ften_d);

  return ret;
}

template <typename TensorType>
Genten::Ktensor
online_gcp_host(const std::vector<TensorType>& X, 
                const TensorType& X0, 
                const Genten::Ktensor& u0,
                Genten::AlgParams& algParams,
                Genten::AlgParams& temporalAlgParams, 
                Genten::AlgParams& spatialAlgParams,
                std::ostream& out,
                Genten::Array& fest,
                Genten::Array& ften)
{
  Genten::Ktensor ret;
  if (algParams.exec_space == Genten::Execution_Space::Default)
    ret = online_gcp_impl<Genten::DefaultExecutionSpace>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
#ifdef HAVE_CUDA
  else if (algParams.exec_space == Genten::Execution_Space::Cuda)
    ret = online_gcp_impl<Kokkos::Cuda>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
#endif
#ifdef HAVE_HIP
  else if (algParams.exec_space == Genten::Execution_Space::HIP)
    ret = online_gcp_impl<Kokkos::Experimental::HIP>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
#endif
#ifdef HAVE_SYCL
  else if (algParams.exec_space == Genten::Execution_Space::SYCL)
    ret = online_gcp_impl<Kokkos::Experimental::SYCL>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
#endif
#ifdef HAVE_OPENMP
  else if (algParams.exec_space == Genten::Execution_Space::OpenMP)
    ret = online_gcp_impl<Kokkos::OpenMP>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
#endif
#ifdef HAVE_THREADS
  else if (algParams.exec_space == Genten::Execution_Space::Threads)
    ret = online_gcp_impl<Kokkos::Threads>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
#endif
#ifdef HAVE_SERIAL
  else if (algParams.exec_space == Genten::Execution_Space::Serial)
    ret = online_gcp_impl<Kokkos::Serial>(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, out, fest, ften);
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

  m.def("driver", [](const Genten::Tensor& x, Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::Ktensor, Genten::PerfHistory > {
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
      return std::make_tuple(u, u0, perfInfo);
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

    Returns a tuple of the Ktensor solution, initial guess, and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));
    m.def("driver", [](const Genten::Sptensor& x, Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::Ktensor, Genten::PerfHistory > {
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
        return std::make_tuple(u, u0, perfInfo);
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

    Returns a tuple of the Ktensor solution, initial guess, and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));

    m.def("driver", [](const Genten::DTC& dtc, const Genten::Tensor& x, Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::Ktensor, Genten::PerfHistory > {
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
        return std::make_tuple(u, u0, perfInfo);
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

    Returns a tuple of the Ktensor solution, initial guess, and PerfHistory containing
    information on the performance of the algorithm.)", pybind11::arg("dtc"), pybind11::arg("X"), pybind11::arg("u0"), pybind11::arg("algParams"));
    m.def("driver", [](const Genten::DTC& dtc, const Genten::Sptensor& x, Genten::Ktensor& u0, Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::Ktensor, Genten::PerfHistory > {
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
        return std::make_tuple(u, u0, perfInfo);
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

    m.def("online_gcp_driver", [](const std::vector<Genten::Sptensor>& X, const Genten::Sptensor& X0, const Genten::Ktensor& u0,
                                  Genten::AlgParams& algParams, Genten::AlgParams& temporalAlgParams, Genten::AlgParams& spatialAlgParams)
                                  -> std::tuple< Genten::Ktensor, std::vector<ttb_real>, std::vector<ttb_real> > {
        py::scoped_ostream_redirect stream(
          std::cout,                                // std::ostream&
          py::module_::import("sys").attr("stdout") // Python output
          );
        py::scoped_ostream_redirect err_stream(
          std::cerr,                                // std::ostream&
          py::module_::import("sys").attr("stderr") // Python output
          );
        Genten::Array fest, ften;
        Genten::Ktensor u = 
          online_gcp_host(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, std::cout, fest, ften);
        std::vector<ttb_real> fest_vec(fest.ptr(),fest.ptr()+fest.size());
        std::vector<ttb_real> ften_vec(ften.ptr(),ften.ptr()+ften.size());
        return std::make_tuple(u, fest_vec, ften_vec);
      }, R"(
    Low-level driver for calling GenTen's streaming GCP method on sparse tensors.

    Users should usually not call this directly, but rather the provided wrapper.

    Parameters:
      * X: list of tensors to compute the CP decomposition from.
      * X0: tensor for computing warm start
      * u0: the initial guess, which may be empty, in which case GenTen will
        compute a random initial guess.
      * algParams: algorithmic parameters controlling the overall streaming method.
      * temporalAlgParams:  algorithmic parameters for temporal GCP solve.
      * spatialAlgParams: algorithmic parameters for the spatial GCP solve.

    Returns a tuple of the Ktensor solution, estimated objective function at each time step, and tensor contribution to the objective.)", 
      pybind11::arg("X"), pybind11::arg("X0"), pybind11::arg("u0"), pybind11::arg("algParams"), pybind11::arg("temporalAlgParams"), pybind11::arg("spatialAlgParams"));

    m.def("online_gcp_driver", [](const std::vector<Genten::Tensor>& X, const Genten::Tensor& X0, const Genten::Ktensor& u0,
                                  Genten::AlgParams& algParams, Genten::AlgParams& temporalAlgParams, Genten::AlgParams& spatialAlgParams)
                                  -> std::tuple< Genten::Ktensor, std::vector<ttb_real>, std::vector<ttb_real> > {
        py::scoped_ostream_redirect stream(
          std::cout,                                // std::ostream&
          py::module_::import("sys").attr("stdout") // Python output
          );
        py::scoped_ostream_redirect err_stream(
          std::cerr,                                // std::ostream&
          py::module_::import("sys").attr("stderr") // Python output
          );
        Genten::Array fest, ften;
        Genten::Ktensor u = 
          online_gcp_host(X, X0, u0, algParams, temporalAlgParams, spatialAlgParams, std::cout, fest, ften);
        std::vector<ttb_real> fest_vec(fest.ptr(),fest.ptr()+fest.size());
        std::vector<ttb_real> ften_vec(ften.ptr(),ften.ptr()+ften.size());
        return std::make_tuple(u, fest_vec, ften_vec);
      }, R"(
    Low-level driver for calling GenTen's streaming GCP method on dense tensors.

    Users should usually not call this directly, but rather the provided wrapper.

    Parameters:
      * X: list of tensors to compute the CP decomposition from.
      * X0: tensor for computing warm start
      * u0: the initial guess, which may be empty, in which case GenTen will
        compute a random initial guess.
      * algParams: algorithmic parameters controlling the overall streaming method.
      * temporalAlgParams:  algorithmic parameters for temporal GCP solve.
      * spatialAlgParams: algorithmic parameters for the spatial GCP solve.

    Returns a tuple of the Ktensor solution, estimated objective function at each time step, and tensor contribution to the objective.)", 
      pybind11::arg("X"), pybind11::arg("X0"), pybind11::arg("u0"), pybind11::arg("algParams"), pybind11::arg("temporalAlgParams"), pybind11::arg("spatialAlgParams"));

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
    m.def("import_sptensor", [](const std::string& fName, const ttb_indx index_base = 0,
                                const bool compressed = false) -> Genten::Sptensor {
        Genten::Sptensor X;
        Genten::import_sptensor(fName, X, index_base, compressed);
        return X;
    }, R"(
    Read and return a sparse tensor from a given file.)", pybind11::arg("file"), pybind11::arg("index_base") = 0, pybind11::arg("compressed") = false);

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
