#include "Genten_Pybind11_classes.hpp"

#include "Genten_PerfHistory.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Driver.hpp"

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

}


namespace py = pybind11;

void pygenten_perfhistory(py::module &m){
    {
        py::class_<Genten::PerfHistory::Entry, std::shared_ptr<Genten::PerfHistory::Entry>> cl(m, "Entry");
        cl.def( py::init( [](){ return new Genten::PerfHistory::Entry(); } ) );
        cl.def_readwrite("iteration", &Genten::PerfHistory::Entry::iteration);
        cl.def_readwrite("residual", &Genten::PerfHistory::Entry::residual);
        cl.def_readwrite("fit", &Genten::PerfHistory::Entry::fit);
        cl.def_readwrite("grad_norm", &Genten::PerfHistory::Entry::grad_norm);
        cl.def_readwrite("cum_time", &Genten::PerfHistory::Entry::cum_time);
        cl.def_readwrite("mttkrp_throughput", &Genten::PerfHistory::Entry::mttkrp_throughput);
        cl.def("__str__", [](Genten::PerfHistory::Entry const &o) -> std::string {
            std::ostringstream s;
            s << o.iteration << " " << o.residual << " " << o.fit << " " << o.grad_norm << " " << o.cum_time << " " << o.mttkrp_throughput;
            return s.str();
        } );
    }
    {
        py::class_<Genten::PerfHistory, std::shared_ptr<Genten::PerfHistory>> cl(m, "PerfHistory");
        cl.def( py::init( [](){ return new Genten::PerfHistory(); } ) );
        cl.def("addEntry", (void (Genten::PerfHistory::*)(const class Genten::PerfHistory::Entry &)) &Genten::PerfHistory::addEntry, "Add a new entry", pybind11::arg("entry"));
        cl.def("addEntry", (void (Genten::PerfHistory::*)()) &Genten::PerfHistory::addEntry, "Add an empty entry");
        cl.def("getEntry", (class Genten::PerfHistory::Entry & (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::getEntry, "Get a given entry", pybind11::arg("i"));
        cl.def("__getitem__", (class Genten::PerfHistory::Entry & (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::getEntry, "Get a given entry", pybind11::arg("i"));
        cl.def("lastEntry", (class Genten::PerfHistory::Entry & (Genten::PerfHistory::*)()) &Genten::PerfHistory::lastEntry, "Get the last entry");
        cl.def("size", (ttb_indx (Genten::PerfHistory::*)()) &Genten::PerfHistory::size, "The number of entries");
        cl.def("resize", (void (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::resize, "Resize to given size", pybind11::arg("new_size"));
        cl.def("__str__", [](Genten::PerfHistory const &o) -> std::string {
            std::ostringstream s;
            o.print(s);
            return s.str();
        } ); 
    }
}

void pygenten_algparams(py::module &m){
    py::enum_<Genten::Execution_Space::type>(m, "Execution_Space")
        .value("Cuda", Genten::Execution_Space::type::Cuda)
        .value("HIP", Genten::Execution_Space::type::HIP)
        .value("SYCL", Genten::Execution_Space::type::SYCL)
        .value("OpenMP", Genten::Execution_Space::type::OpenMP)
        .value("Threads", Genten::Execution_Space::type::Threads)
        .value("Serial", Genten::Execution_Space::type::Serial)
        .value("Default", Genten::Execution_Space::type::Default)
        .export_values();
    py::enum_<Genten::Solver_Method::type>(m, "Solver_Method")
        .value("CP_ALS", Genten::Solver_Method::type::CP_ALS)
        .value("CP_OPT", Genten::Solver_Method::type::CP_OPT)
        .value("GCP_SGD", Genten::Solver_Method::type::GCP_SGD)
        .value("GCP_SGD_DIST", Genten::Solver_Method::type::GCP_SGD_DIST)
        .value("GCP_OPT", Genten::Solver_Method::type::GCP_OPT)
        .export_values();
    py::enum_<Genten::Opt_Method::type>(m, "Opt_Method")
        .value("LBFGSB", Genten::Opt_Method::type::LBFGSB)
        .value("ROL", Genten::Opt_Method::type::ROL)
        .export_values();
    py::enum_<Genten::MTTKRP_Method::type>(m, "MTTKRP_Method")
        .value("Default", Genten::MTTKRP_Method::type::Default)
        .value("OrigKokkos", Genten::MTTKRP_Method::type::OrigKokkos)
        .value("Atomic", Genten::MTTKRP_Method::type::Atomic)
        .value("Duplicated", Genten::MTTKRP_Method::type::Duplicated)
        .value("Single", Genten::MTTKRP_Method::type::Single)
        .value("Perm", Genten::MTTKRP_Method::type::Perm)
        .export_values();
    py::enum_<Genten::MTTKRP_All_Method::type>(m, "MTTKRP_All_Method")
        .value("Default", Genten::MTTKRP_All_Method::type::Default)
        .value("Iterated", Genten::MTTKRP_All_Method::type::Iterated)
        .value("Atomic", Genten::MTTKRP_All_Method::type::Atomic)
        .value("Duplicated", Genten::MTTKRP_All_Method::type::Duplicated)
        .value("Single", Genten::MTTKRP_All_Method::type::Single)
        .export_values();
    py::enum_<Genten::Dist_Update_Method::type>(m, "Dist_Update_Method")
        .value("AllReduce", Genten::Dist_Update_Method::type::AllReduce)
        .value("AllGather", Genten::Dist_Update_Method::type::AllGather)
        .value("Tpetra", Genten::Dist_Update_Method::type::Tpetra)
        .export_values();
    py::enum_<Genten::Hess_Vec_Method::type>(m, "Hess_Vec_Method")
        .value("Full", Genten::Hess_Vec_Method::type::Full)
        .value("GaussNewton", Genten::Hess_Vec_Method::type::GaussNewton)
        .value("FiniteDifference", Genten::Hess_Vec_Method::type::FiniteDifference)
        .export_values();
    py::enum_<Genten::Hess_Vec_Tensor_Method::type>(m, "Hess_Vec_Tensor_Method")
        .value("Default", Genten::Hess_Vec_Tensor_Method::type::Default)
        .value("Atomic", Genten::Hess_Vec_Tensor_Method::type::Atomic)
        .value("Duplicated", Genten::Hess_Vec_Tensor_Method::type::Duplicated)
        .value("Single", Genten::Hess_Vec_Tensor_Method::type::Single)
        .value("Perm", Genten::Hess_Vec_Tensor_Method::type::Perm)
        .export_values();
    py::enum_<Genten::Hess_Vec_Prec_Method::type>(m, "Hess_Vec_Prec_Method")
        .value("None", Genten::Hess_Vec_Prec_Method::type::None)
        .value("ApproxBlockDiag", Genten::Hess_Vec_Prec_Method::type::ApproxBlockDiag)
        .export_values();
    py::enum_<Genten::TTM_Method::type>(m, "TTM_Method")
        .value("DGEMM", Genten::TTM_Method::type::DGEMM)
        .value("Parfor_DGEMM", Genten::TTM_Method::type::Parfor_DGEMM)
        .export_values();
    py::enum_<Genten::GCP_LossFunction::type>(m, "GCP_LossFunction")
        .value("Gaussian", Genten::GCP_LossFunction::type::Gaussian)
        .value("Rayleigh", Genten::GCP_LossFunction::type::Rayleigh)
        .value("Gamma", Genten::GCP_LossFunction::type::Gamma)
        .value("Bernoulli", Genten::GCP_LossFunction::type::Bernoulli)
        .value("Poisson", Genten::GCP_LossFunction::type::Poisson)
        .export_values();
    py::enum_<Genten::GCP_Sampling::type>(m, "GCP_Sampling")
        .value("Uniform", Genten::GCP_Sampling::type::Uniform)
        .value("Stratified", Genten::GCP_Sampling::type::Stratified)
        .value("SemiStratified", Genten::GCP_Sampling::type::SemiStratified)
        .value("Dense", Genten::GCP_Sampling::type::Dense)
        .export_values();
    py::enum_<Genten::GCP_Step::type>(m, "GCP_Step")
        .value("SGD", Genten::GCP_Step::type::SGD)
        .value("ADAM", Genten::GCP_Step::type::ADAM)
        .value("AdaGrad", Genten::GCP_Step::type::AdaGrad)
        .value("AMSGrad", Genten::GCP_Step::type::AMSGrad)
        .value("SGDMomentum", Genten::GCP_Step::type::SGDMomentum)
        .value("DEMON", Genten::GCP_Step::type::DEMON)
        .export_values();
    py::enum_<Genten::GCP_Streaming_Solver::type>(m, "GCP_Streaming_Solver")
        .value("SGD", Genten::GCP_Streaming_Solver::type::SGD)
        .value("LeastSquares", Genten::GCP_Streaming_Solver::type::LeastSquares)
        .value("OnlineCP", Genten::GCP_Streaming_Solver::type::OnlineCP)
        .export_values();
    py::enum_<Genten::GCP_Streaming_Window_Method::type>(m, "GCP_Streaming_Window_Method")
        .value("Reservoir", Genten::GCP_Streaming_Window_Method::type::Reservoir)
        .value("Last", Genten::GCP_Streaming_Window_Method::type::Last)
        .export_values();
    py::enum_<Genten::GCP_Streaming_History_Method::type>(m, "GCP_Streaming_History_Method")
        .value("Ktensor_Fro", Genten::GCP_Streaming_History_Method::type::Ktensor_Fro)
        .value("Factor_Fro", Genten::GCP_Streaming_History_Method::type::Factor_Fro)
        .value("GCP_Loss", Genten::GCP_Streaming_History_Method::type::GCP_Loss)
        .export_values();
    {
        py::class_<Genten::AlgParams, std::shared_ptr<Genten::AlgParams>> cl(m, "AlgParams");
        cl.def( py::init( [](){ return new Genten::AlgParams(); } ) );
        cl.def_readwrite("exec_space", &Genten::AlgParams::exec_space);
        cl.def_readwrite("method", &Genten::AlgParams::method);
        cl.def_readwrite("rank", &Genten::AlgParams::rank);
        cl.def_readwrite("seed", &Genten::AlgParams::seed);
        cl.def_readwrite("prng", &Genten::AlgParams::prng);
        cl.def_readwrite("maxiters", &Genten::AlgParams::maxiters);
        cl.def_readwrite("maxsecs", &Genten::AlgParams::maxsecs);
        cl.def_readwrite("tol", &Genten::AlgParams::tol);
        cl.def_readwrite("printitn", &Genten::AlgParams::printitn);
        cl.def_readwrite("debug", &Genten::AlgParams::debug);
        cl.def_readwrite("timings", &Genten::AlgParams::timings);
        cl.def_readwrite("full_gram", &Genten::AlgParams::full_gram);
        cl.def_readwrite("rank_def_solver", &Genten::AlgParams::rank_def_solver);
        cl.def_readwrite("rcond", &Genten::AlgParams::rcond);
        cl.def_readwrite("penalty", &Genten::AlgParams::penalty);
        cl.def_readwrite("dist_guess_method", &Genten::AlgParams::dist_guess_method);
        cl.def_readwrite("scale_guess_by_norm_x", &Genten::AlgParams::scale_guess_by_norm_x);

        cl.def_readwrite("mttkrp_method", &Genten::AlgParams::mttkrp_method);
        cl.def_readwrite("mttkrp_all_method", &Genten::AlgParams::mttkrp_all_method);
        cl.def_readwrite("mttkrp_nnz_tile_size", &Genten::AlgParams::mttkrp_nnz_tile_size);
        cl.def_readwrite("mttkrp_duplicated_factor_matrix_tile_size", &Genten::AlgParams::mttkrp_duplicated_factor_matrix_tile_size);
        cl.def_readwrite("mttkrp_duplicated_threshold", &Genten::AlgParams::mttkrp_duplicated_threshold);
        cl.def_readwrite("dist_update_method", &Genten::AlgParams::dist_update_method);
        cl.def_readwrite("optimize_maps", &Genten::AlgParams::optimize_maps);
        cl.def_readwrite("build_maps_on_device", &Genten::AlgParams::build_maps_on_device);
        cl.def_readwrite("warmup", &Genten::AlgParams::warmup);

        cl.def_readwrite("ttm_method", &Genten::AlgParams::ttm_method);

        cl.def_readwrite("opt_method", &Genten::AlgParams::opt_method);
        cl.def_readwrite("lower", &Genten::AlgParams::lower);
        cl.def_readwrite("upper", &Genten::AlgParams::upper);
        cl.def_readwrite("rolfilename", &Genten::AlgParams::rolfilename);
        cl.def_readwrite("factr", &Genten::AlgParams::factr);
        cl.def_readwrite("pgtol", &Genten::AlgParams::pgtol);
        cl.def_readwrite("memory", &Genten::AlgParams::memory);
        cl.def_readwrite("max_total_iters", &Genten::AlgParams::max_total_iters);
        cl.def_readwrite("hess_vec_method", &Genten::AlgParams::hess_vec_method);
        cl.def_readwrite("hess_vec_tensor_method", &Genten::AlgParams::hess_vec_tensor_method);
        cl.def_readwrite("hess_vec_prec_method", &Genten::AlgParams::hess_vec_prec_method);

        cl.def_readwrite("loss_function_type", &Genten::AlgParams::loss_function_type);
        cl.def_readwrite("loss_eps", &Genten::AlgParams::loss_eps);
        cl.def_readwrite("gcp_tol", &Genten::AlgParams::gcp_tol);

        cl.def_readwrite("sampling_type", &Genten::AlgParams::sampling_type);
        cl.def_readwrite("rate", &Genten::AlgParams::rate);
        cl.def_readwrite("decay", &Genten::AlgParams::decay);
        cl.def_readwrite("max_fails", &Genten::AlgParams::max_fails);
        cl.def_readwrite("epoch_iters", &Genten::AlgParams::epoch_iters);
        cl.def_readwrite("frozen_iters", &Genten::AlgParams::frozen_iters);
        cl.def_readwrite("rng_iters", &Genten::AlgParams::rng_iters);
        cl.def_readwrite("gcp_seed", &Genten::AlgParams::gcp_seed);
        cl.def_readwrite("num_samples_nonzeros_value", &Genten::AlgParams::num_samples_nonzeros_value);
        cl.def_readwrite("num_samples_zeros_value", &Genten::AlgParams::num_samples_zeros_value);
        cl.def_readwrite("num_samples_nonzeros_grad", &Genten::AlgParams::num_samples_nonzeros_grad);
        cl.def_readwrite("num_samples_zeros_grad", &Genten::AlgParams::num_samples_zeros_grad);
        cl.def_readwrite("oversample_factor", &Genten::AlgParams::oversample_factor);
        cl.def_readwrite("bulk_factor", &Genten::AlgParams::bulk_factor);
        cl.def_readwrite("w_f_nz", &Genten::AlgParams::w_f_nz);
        cl.def_readwrite("w_f_z", &Genten::AlgParams::w_f_z);
        cl.def_readwrite("w_g_nz", &Genten::AlgParams::w_g_nz);
        cl.def_readwrite("w_g_z", &Genten::AlgParams::w_g_z);
        cl.def_readwrite("normalize", &Genten::AlgParams::normalize);
        cl.def_readwrite("hash", &Genten::AlgParams::hash);
        cl.def_readwrite("fuse", &Genten::AlgParams::fuse);
        cl.def_readwrite("fuse_sa", &Genten::AlgParams::fuse_sa);
        cl.def_readwrite("compute_fit", &Genten::AlgParams::compute_fit);
        cl.def_readwrite("step_type", &Genten::AlgParams::step_type);
        cl.def_readwrite("adam_beta1", &Genten::AlgParams::adam_beta1);
        cl.def_readwrite("adam_beta2", &Genten::AlgParams::adam_beta2);
        cl.def_readwrite("adam_eps", &Genten::AlgParams::adam_eps);
        cl.def_readwrite("async", &Genten::AlgParams::async);
        cl.def_readwrite("anneal", &Genten::AlgParams::anneal);
        cl.def_readwrite("anneal_min_lr", &Genten::AlgParams::anneal_min_lr);
        cl.def_readwrite("anneal_max_lr", &Genten::AlgParams::anneal_max_lr);

        cl.def_readwrite("streaming_solver", &Genten::AlgParams::streaming_solver);
        cl.def_readwrite("history_method", &Genten::AlgParams::history_method);
        cl.def_readwrite("window_method", &Genten::AlgParams::window_method);
        cl.def_readwrite("window_size", &Genten::AlgParams::window_size);
        cl.def_readwrite("window_weight", &Genten::AlgParams::window_weight);
        cl.def_readwrite("window_penalty", &Genten::AlgParams::window_penalty);
        cl.def_readwrite("factor_penalty", &Genten::AlgParams::factor_penalty);

        cl.def("__str__", [](Genten::AlgParams const &o) -> std::string {
            std::ostringstream s;
            o.print(s);
            return s.str();
        } );
    }    
}

void pygenten_ktensor(pybind11::module &m){
    {
        py::class_<Genten::RandomMT, std::shared_ptr<Genten::RandomMT>> cl(m, "RandomMT");
        cl.def( py::init( [](const unsigned long  nnSeed){ return new Genten::RandomMT(nnSeed); } ) );
        cl.def("genrnd_int32", (unsigned long (Genten::RandomMT::*)()) &Genten::RandomMT::genrnd_int32, "Return a uniform random number on the interval [0,0xffffffff].");
        cl.def("genrnd_double", (double(Genten::RandomMT::*)()) &Genten::RandomMT::genrnd_double, "Return a uniform random number on the interval [0,1).");
        cl.def("genrnd_doubleInclusive", (double(Genten::RandomMT::*)()) &Genten::RandomMT::genrnd_doubleInclusive, "Return a uniform random number on the interval [0,1].");
        cl.def("genMatlabMT", (double(Genten::RandomMT::*)()) &Genten::RandomMT::genMatlabMT, "Return a uniform random number on the interval [0,1).");
    }
    {
        py::class_<Genten::IndxArray, std::shared_ptr<Genten::IndxArray>> cl(m, "IndxArray", py::buffer_protocol());
        cl.def( py::init( [](){ return new Genten::IndxArray(); } ) );
        cl.def( py::init( [](ttb_indx n){ return new Genten::IndxArray(n); } ) );
        cl.def( py::init( [](ttb_indx n, ttb_indx val){ return new Genten::IndxArray(n, val); } ) );
        cl.def( py::init( [](ttb_indx n, ttb_indx *v){ return new Genten::IndxArray(n, v); } ) );
        cl.def( py::init( [](ttb_indx n, const ttb_real *v, bool subtract_one){ return new Genten::IndxArray(n, v, subtract_one); } ) );
        cl.def_buffer([](Genten::IndxArray &m) -> py::buffer_info {
                return py::buffer_info(
                    &m[0], sizeof(ttb_indx), py::format_descriptor<ttb_indx>::format(), m.size()
                );
            });
        cl.def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<ttb_indx>::format())
                throw std::runtime_error("Incompatible format: expected a ttb_indx array!");

            if (info.ndim != 1)
                throw std::runtime_error("Incompatible buffer dimension!");

            return Genten::IndxArray(info.shape[0], static_cast<ttb_indx *>(info.ptr));
        }));
    }
    {
        py::class_<Genten::Array, std::shared_ptr<Genten::Array>> cl(m, "Array", py::buffer_protocol());
        cl.def( py::init( [](){ return new Genten::Array(); } ) );
        cl.def( py::init( [](ttb_indx n){ return new Genten::Array(n); } ) );
        cl.def( py::init( [](ttb_indx n, bool parallel){ return new Genten::Array(n, parallel); } ) );
        cl.def( py::init( [](ttb_indx n, ttb_real val){ return new Genten::Array(n, val); } ) );
        cl.def( py::init( [](ttb_indx n, ttb_real *d){ return new Genten::Array(n, d); } ) );
        cl.def( py::init( [](ttb_indx n, ttb_real *d, bool shdw){ return new Genten::Array(n, d, shdw); } ) );
        cl.def( py::init( [](const Genten::Array & src){ return new Genten::Array(src); } ) );
        cl.def_buffer([](Genten::Array &m) -> py::buffer_info {
                return py::buffer_info(
                    &m[0], sizeof(ttb_real), py::format_descriptor<ttb_real>::format(), m.size()
                );
            });
        cl.def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<ttb_real>::format())
                throw std::runtime_error("Incompatible format: expected a ttb_real array!");

            if (info.ndim != 1)
                throw std::runtime_error("Incompatible buffer dimension!");

            // Always use false here so we don't alias python arrays (they
            // may be temporaries)
            return Genten::Array(info.shape[0], static_cast<ttb_real *>(info.ptr), false);
        }));
    }
    {
        py::class_<Genten::FacMatrix, std::shared_ptr<Genten::FacMatrix>> cl(m, "FacMatrix", py::buffer_protocol());
        cl.def( py::init( [](){ return new Genten::FacMatrix(); } ) );
        cl.def( py::init( [](ttb_indx m, ttb_indx n){ return new Genten::FacMatrix(m, n); } ) );
        cl.def( py::init( [](ttb_indx m, ttb_indx n, const ttb_real * cvec){ return new Genten::FacMatrix(m, n, cvec); } ) );
        cl.def_buffer([](Genten::FacMatrix &m) -> py::buffer_info {
                return py::buffer_info(
                    m.rowptr(0),
                    sizeof(ttb_real),
                    py::format_descriptor<ttb_real>::format(),
                    2,
                    { m.nRows(), m.nCols() },
                    { m.nCols() * sizeof(ttb_real), sizeof(ttb_real)}
                );
            });
        cl.def(py::init([](py::buffer b) {
            py::buffer_info info = b.request();

            if (info.format != py::format_descriptor<ttb_real>::format())
                throw std::runtime_error("Incompatible format: expected a ttb_real array!");

            if (info.ndim != 2)
                throw std::runtime_error("Incompatible buffer dimension!");

            return Genten::FacMatrix(info.shape[0], info.shape[1], static_cast<ttb_real *>(info.ptr));
        }));
    }
    {
        py::class_<Genten::FacMatArray, std::shared_ptr<Genten::FacMatArray>> cl(m, "FacMatArray");
        cl.def( py::init( [](ttb_indx n){ return new Genten::FacMatArray(n); } ) );
        cl.def( py::init( [](ttb_indx n, const Genten::IndxArray & nrow, ttb_indx ncol){ return new Genten::FacMatArray(n, nrow, ncol); } ) );
        cl.def( py::init( [](const Genten::FacMatArray & src){ return new Genten::FacMatArray(src); } ) );
        cl.def("size", (ttb_indx (Genten::FacMatArray::*)()) &Genten::FacMatArray::size, "Return the number of factor matrices.");
        cl.def("reals", (ttb_indx (Genten::FacMatArray::*)()) &Genten::FacMatArray::reals, "Count the total ttb_reals currently stored here for any purpose.");
        cl.def("set_factor", (void (Genten::FacMatArray::*)(const ttb_indx, const Genten::FacMatrix&)) &Genten::FacMatArray::set_factor, "Set a factor matrix");
    }
    {
        py::class_<Genten::Ktensor, std::shared_ptr<Genten::Ktensor>> cl(m, "Ktensor");
        cl.def( py::init( [](){ return new Genten::Ktensor(); } ) );
        cl.def( py::init( [](ttb_indx nc, ttb_indx nd){ return new Genten::Ktensor(nc, nd); } ), "" , pybind11::arg("nc"), pybind11::arg("nd"));
        cl.def( py::init( [](ttb_indx nc, ttb_indx nd, const Genten::IndxArray &sz){ return new Genten::Ktensor(nc, nd, sz); } ), "" , pybind11::arg("nc"), pybind11::arg("nd"), pybind11::arg("sz"));
        cl.def( py::init( [](const Genten::Array &w, const Genten::FacMatArray &vals){ return new Genten::Ktensor(w, vals); } ), "" , pybind11::arg("w"), pybind11::arg("vals"));
        cl.def("setWeightsRand", (void (Genten::Ktensor::*)()) &Genten::Ktensor::setWeightsRand, "Set all entries to random values between 0 and 1.  Does not change the matrix array, so the Ktensor can become inconsistent");
        cl.def("setWeights", [](Genten::Ktensor const &o, ttb_real val) -> void { o.setWeights(val); }, "Set all weights equal to val.", pybind11::arg("val"));
        cl.def("setWeights", [](Genten::Ktensor const &o, const Genten::Array &newWeights) -> void { o.setWeights(newWeights); }, "Set all weights equal to val.", pybind11::arg("newWeights"));
        cl.def("setMatrices", (void (Genten::Ktensor::*)(ttb_real)) &Genten::Ktensor::setMatrices, "Set all matrix entries equal to val.", pybind11::arg("val"));
        cl.def("setMatricesRand", (void (Genten::Ktensor::*)()) &Genten::Ktensor::setMatricesRand, "Set all entries to random values in [0,1).");
        cl.def("setMatricesScatter", (void (Genten::Ktensor::*)(const bool, const bool, Genten::RandomMT &)) &Genten::Ktensor::setMatricesScatter, "Set all entries to reproducible random values.", pybind11::arg("bUseMatlabRNG"), pybind11::arg("bUseParallelRNG"), pybind11::arg("cRMT"));
        cl.def("setRandomUniform", (void (Genten::Ktensor::*)(const bool, Genten::RandomMT &)) &Genten::Ktensor::setRandomUniform, "Fill the Ktensor with uniform random values, normalized to be stochastic.", pybind11::arg("bUseMatlabRNG"), pybind11::arg("cRMT"));
        cl.def("scaleRandomElements", (void (Genten::Ktensor::*)()) &Genten::Ktensor::scaleRandomElements, "multiply (plump) a fraction (indices randomly chosen) of each FacMatrix by scale.");
        //setProcessorMap - ProcessorMap
        //getProcessorMap - ProcessorMap
        cl.def("ncomponents", (ttb_indx (Genten::Ktensor::*)()) &Genten::Ktensor::ncomponents, "Return number of components.");
        cl.def("ndims", (ttb_indx (Genten::Ktensor::*)()) &Genten::Ktensor::ndims, "Return number of dimensions of Ktensor.");
        cl.def("isConsistent", [](Genten::Ktensor const &o) -> bool { return o.isConsistent(); }, "Consistency check on sizes.");
        cl.def("isConsistent", [](Genten::Ktensor const &o, const Genten::IndxArray & sz) -> bool { return o.isConsistent(sz); }, "Consistency check on sizes.");
        cl.def("hasNonFinite", (bool (Genten::Ktensor::*)(ttb_indx &)) &Genten::Ktensor::hasNonFinite, "", pybind11::arg("bad"));
        cl.def("isNonnegative", (bool (Genten::Ktensor::*)(bool)) &Genten::Ktensor::isNonnegative, "", pybind11::arg("bDisplayErrors"));
        cl.def("weights", [](Genten::Ktensor const &o) -> Genten::Array { return o.weights(); }, "Return reference to weights vector.");
        cl.def("weights", [](Genten::Ktensor const &o, ttb_indx i) -> ttb_real { return o.weights(i); }, "Return reference to weights vector.", pybind11::arg("i"));
        cl.def("__getitem__", [](Genten::Ktensor const &o, ttb_indx n) -> const Genten::FacMatrix & { return o[n]; }, "Return a reference to the n-th factor matrix", pybind11::arg("n"));
    }
    {
        py::class_<Genten::Tensor, std::shared_ptr<Genten::Tensor>> cl(m, "Tensor");
        cl.def( py::init( [](){ return new Genten::Tensor(); } ), "Empty constructor" );
        cl.def( py::init( [](const Genten::Tensor& src){ return new Genten::Tensor(src); } ), "Copy constructor", pybind11::arg("src") );
        cl.def( py::init( [](const Genten::IndxArray &sz){ return new Genten::Tensor(sz); } ), "Construct tensor of given size initialized to val", pybind11::arg("sz"));
        cl.def( py::init( [](const Genten::IndxArray &sz, ttb_real val){ return new Genten::Tensor(sz, val); } ), "Construct tensor of given size initialized to val", pybind11::arg("sz"), pybind11::arg("val"));
        cl.def( py::init( [](const Genten::IndxArray &sz, const Genten::Array &vals){ return new Genten::Tensor(sz, vals); } ), "Construct tensor with given size and values", pybind11::arg("sz"), pybind11::arg("vals"));
        cl.def("ndims", (ttb_indx (Genten::Tensor::*)()) &Genten::Tensor::ndims, "Return the number of dimensions (i.e., the order).");
        cl.def("size", [](Genten::Tensor const &o, ttb_indx i) -> ttb_indx { return o.size(i); } , "Return size of dimension i.", pybind11::arg("i"));
        cl.def("size", [](Genten::Tensor const &o) -> Genten::IndxArray { return o.size(); } , "Return sizes array.");
        cl.def("numel", (ttb_indx (Genten::Tensor::*)()) &Genten::Tensor::numel, "Return the total number of elements in the tensor.");
    }

    m.def("driver", [](const Genten::Tensor& x,
                       const Genten::Ktensor& u0,
                       Genten::AlgParams& algParams) -> std::tuple< Genten::Ktensor, Genten::PerfHistory > {
       Genten::ptree ptree;
       Genten::PerfHistory perfInfo;
       Genten::Ktensor u = driver_host(x, u0, algParams, ptree, perfInfo, std::cout);
       return std::make_tuple(u, perfInfo);
    });

    m.def("import_tensor", [](const std::string& fName) -> Genten::Tensor {
        Genten::Tensor X;
        Genten::import_tensor(fName, X);
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
}

void pygenten_classes(pybind11::module &m){
    pygenten_perfhistory(m);
    pygenten_algparams(m);
    pygenten_ktensor(m);
}
