#include "Genten_Pybind11_classes.hpp"

#include "Genten_PerfHistory.hpp"
#include "Genten_AlgParams.hpp"
#include "Genten_Ktensor.hpp"
#include "Genten_Tensor.hpp"
#include "Genten_Sptensor.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Pmap.hpp"
#include "Genten_DistTensorContext.hpp"

#include <pybind11/iostream.h>

namespace py = pybind11;
namespace Genten {
  using DTC = DistTensorContext<Genten::DefaultHostExecutionSpace>;
}

namespace {

  Genten::IndxArray
  make_indxarray(py::buffer b) {
    Genten::IndxArray a;
    py::buffer_info info = b.request();
    if (info.ndim != 1)
      throw std::runtime_error("Incompatible buffer dimension!");
    const ttb_indx n = info.shape[0];
    if (info.format == py::format_descriptor<ttb_indx>::format())
      a = Genten::IndxArray(n, static_cast<ttb_indx*>(info.ptr));
    else if (info.format == py::format_descriptor<ttb_real>::format())
      a = Genten::IndxArray(n, static_cast<ttb_real*>(info.ptr));
    else
      throw std::runtime_error("Incompatible format: expected a ttb_indx or ttb_real array!  Format is " + info.format);
    return a;
  }

  Genten::IndxArray
  make_indxarray(py::tuple b) {
    const ttb_indx n = b.size();
    Genten::IndxArray a(n);
    for (ttb_indx i=0; i<n; ++i)
      a[i] = py::cast<int>(b[i]);
    return a;
  }

}

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
    cl.def("addEntry", (void (Genten::PerfHistory::*)(const class Genten::PerfHistory::Entry &)) &Genten::PerfHistory::addEntry, "Add a new entry", py::arg("entry"));
    cl.def("addEntry", (void (Genten::PerfHistory::*)()) &Genten::PerfHistory::addEntry, "Add an empty entry");
    cl.def("getEntry", (class Genten::PerfHistory::Entry & (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::getEntry, "Get a given entry", py::arg("i"));
    cl.def("__getitem__", (class Genten::PerfHistory::Entry & (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::getEntry, "Get a given entry", py::arg("i"));
    cl.def("lastEntry", (class Genten::PerfHistory::Entry & (Genten::PerfHistory::*)()) &Genten::PerfHistory::lastEntry, "Get the last entry");
    cl.def("size", (ttb_indx (Genten::PerfHistory::*)()) &Genten::PerfHistory::size, "The number of entries");
    cl.def("resize", (void (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::resize, "Resize to given size", py::arg("new_size"));
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
    .value("ExecSpaceDefault", Genten::Execution_Space::type::Default)
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
    .value("MttkrpDefault", Genten::MTTKRP_Method::type::Default)
    .value("MttkrpOrigKokkos", Genten::MTTKRP_Method::type::OrigKokkos)
    .value("MttkrpAtomic", Genten::MTTKRP_Method::type::Atomic)
    .value("MttkrpDuplicated", Genten::MTTKRP_Method::type::Duplicated)
    .value("MttkrpSingle", Genten::MTTKRP_Method::type::Single)
    .value("MttkrpPerm", Genten::MTTKRP_Method::type::Perm)
    .export_values();
  py::enum_<Genten::MTTKRP_All_Method::type>(m, "MTTKRP_All_Method")
    .value("MttkrpAllDefault", Genten::MTTKRP_All_Method::type::Default)
    .value("MttkrpAllIterated", Genten::MTTKRP_All_Method::type::Iterated)
    .value("AMttkrpAlltomic", Genten::MTTKRP_All_Method::type::Atomic)
    .value("MttkrpAllDuplicated", Genten::MTTKRP_All_Method::type::Duplicated)
    .value("MttkrpAllSingle", Genten::MTTKRP_All_Method::type::Single)
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
    .value("HessVecDefault", Genten::Hess_Vec_Tensor_Method::type::Default)
    .value("HessVecAtomic", Genten::Hess_Vec_Tensor_Method::type::Atomic)
    .value("HessVecDuplicated", Genten::Hess_Vec_Tensor_Method::type::Duplicated)
    .value("HessVecSingle", Genten::Hess_Vec_Tensor_Method::type::Single)
    .value("HessVecPerm", Genten::Hess_Vec_Tensor_Method::type::Perm)
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
    .value("StreamingSGD", Genten::GCP_Streaming_Solver::type::SGD)
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
  py::enum_<Genten::GCP_Goal_Method::type>(m, "GCP_Goal_Method")
    .value("NoGoal", Genten::GCP_Goal_Method::type::None)
    .value("PythonModule", Genten::GCP_Goal_Method::type::PythonModule)
    .value("PythonObject", Genten::GCP_Goal_Method::type::PythonObject)
    .export_values();
  py::enum_<Genten::ProcessorMap::MpiOp>(m, "MpiOp")
    .value("SumOp", Genten::ProcessorMap::MpiOp::Sum)
    .value("MaxOp", Genten::ProcessorMap::MpiOp::Max)
    .export_values();
  {
    py::class_<Genten::AlgParams, std::shared_ptr<Genten::AlgParams>> cl(m, "AlgParams");
    cl.def( py::init( [](){ return new Genten::AlgParams(); } ) );
    cl.def_readwrite("exec_space", &Genten::AlgParams::exec_space);
    // Setting the processor grid this way appears to not work.  Use the
    // set_proc_grid() method below
    //cl.def_readwrite("proc_grid", &Genten::AlgParams::proc_grid);
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
    cl.def_readwrite("gcp_goal_method", &Genten::AlgParams::goal_method);
    cl.def_readwrite("gcp_goal_python_module_name", &Genten::AlgParams::python_module_name);
    cl.def_readwrite("gcp_goal_python_object_name", &Genten::AlgParams::python_object_name);
    // Setting the python object this way appears to not work.  Use the
    // set_py_goal() method below
    //cl.def_readwrite("gcp_goal_python_object", &Genten::AlgParams::python_object);

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

    cl.def("set_proc_grid", [](Genten::AlgParams& a, py::buffer b) {
        a.proc_grid = make_indxarray(b);
      });
    cl.def("set_proc_grid", [](Genten::AlgParams& a, py::tuple b) {
        a.proc_grid = make_indxarray(b);
      });
    cl.def("set_py_goal", [](Genten::AlgParams& a, const py::object& po) {
        a.python_object = po;
        a.goal_method = Genten::GCP_Goal_Method::PythonObject;
      });
  }
}

void pygenten_proc_map(py::module &m){
  {
    py::class_<Genten::ProcessorMap, std::shared_ptr<Genten::ProcessorMap>> cl(m, "ProcessorMap");
    cl.def("gridSize", (ttb_indx (Genten::ProcessorMap::*)()) &Genten::ProcessorMap::gridSize, "Return number of processors in grid");
    cl.def("gridRank", (ttb_indx (Genten::ProcessorMap::*)()) &Genten::ProcessorMap::gridRank, "Return rank of this processor in grid");
    cl.def("gridAllReduceArray", [](const Genten::ProcessorMap& pmap, py::buffer b, Genten::ProcessorMap::MpiOp op) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<ttb_real>::format())
          throw std::runtime_error("Incompatible format: expected a ttb_real array!");
        if (info.ndim != 1)
          throw std::runtime_error("Incompatible buffer dimension!");
        ttb_real *ptr = static_cast<ttb_real*>(info.ptr);
        ttb_indx n = info.shape[0];
        pmap.gridAllReduce(ptr, n, op);
      });
    cl.def("gridAllReduceArray", [](const Genten::ProcessorMap& pmap, py::buffer b) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<ttb_real>::format())
          throw std::runtime_error("Incompatible format: expected a ttb_real array!");
        if (info.ndim != 1)
          throw std::runtime_error("Incompatible buffer dimension!");
        ttb_real *ptr = static_cast<ttb_real*>(info.ptr);
        ttb_indx n = info.shape[0];
        pmap.gridAllReduce(ptr, n);
      });
    cl.def("gridAllReduce", [](const Genten::ProcessorMap& pmap, ttb_real x, Genten::ProcessorMap::MpiOp op) {
        return pmap.gridAllReduce(x, op);
      });
    cl.def("gridAllReduce", [](const Genten::ProcessorMap& pmap, ttb_real x) {
        return pmap.gridAllReduce(x);
      });
  }
}

void pygenten_ktensor(py::module &m){
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
    cl.def(py::init([](py::buffer b) { return make_indxarray(b); }));
    cl.def_buffer([](Genten::IndxArray &m) -> py::buffer_info {
        return py::buffer_info(
          &m[0], sizeof(ttb_real), py::format_descriptor<ttb_indx>::format(), m.size()
          );
      });
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
          { m.view().stride(0)*sizeof(ttb_real), m.view().stride(1)*sizeof(ttb_real)}
          );
      });
    cl.def(py::init([](const py::array_t<ttb_real,py::array::c_style>& b) {
          py::buffer_info info = b.request();
          if (info.ndim != 2)
            throw std::runtime_error("Incompatible buffer dimension!");
          const ttb_indx nrow = info.shape[0];
          const ttb_indx ncol = info.shape[1];
          const ttb_indx s0 = info.strides[0]/sizeof(ttb_real);
          const ttb_indx s1 = info.strides[1]/sizeof(ttb_real);
          if (s1 != 1) {
            std::string msg = "Buffer is not layout-right!  Dims = (" +
              std::to_string(nrow) + ", " + std::to_string(ncol) +
              "), strides = (" + std::to_string(s0) + ", " +
              std::to_string(s1) + ")";
            throw std::runtime_error(msg);
          }
          ttb_real *ptr = static_cast<ttb_real *>(info.ptr);
          Genten::FacMatrix A;
          if (s0 == ncol) {
            Kokkos::View<ttb_real**, Kokkos::LayoutRight, Genten::DefaultHostExecutionSpace> v(ptr, nrow, ncol);
            //A = Genten::FacMatrix(nrow, ncol, v);
            A = Genten::FacMatrix(nrow, ncol);
            deep_copy(A.view(), v);
          }
          else {
            Kokkos::LayoutStride layout;
            layout.dimension[0] = nrow;
            layout.dimension[1] = ncol;
            layout.stride[0] = s0;
            layout.stride[1] = s1;
            Kokkos::View<ttb_real**, Kokkos::LayoutStride, Genten::DefaultHostExecutionSpace> v(ptr, layout);
            //A = Genten::FacMatrix(nrow, ncol, v);
            A = Genten::FacMatrix(nrow, ncol);
            deep_copy(A.view(), v);
          }
          return A;
        }));
    cl.def("__str__", [](const Genten::FacMatrix& A) {
        std::stringstream ss;
        Genten::print_matrix(A, ss);
        return ss.str();
      });
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
    // cl.def( py::init( [](ttb_indx nc, ttb_indx nd){ return new Genten::Ktensor(nc, nd); } ), "" , py::arg("nc"), py::arg("nd"));
    // cl.def( py::init( [](ttb_indx nc, ttb_indx nd, const Genten::IndxArray &sz){ return new Genten::Ktensor(nc, nd, sz); } ), "" , py::arg("nc"), py::arg("nd"), py::arg("sz"));
    // cl.def( py::init( [](const Genten::Array &w, const Genten::FacMatArray &vals){ return new Genten::Ktensor(w, vals); } ), "" , py::arg("w"), py::arg("vals"));

    cl.def(py::init([](const py::array_t<ttb_real>& w, const py::list& f, const bool copy=true) {
          // Get weights
          py::buffer_info w_info = w.request();
          if (w_info.ndim != 1)
            throw std::runtime_error("Incompatible buffer dimension!");
          const ttb_indx nc = w_info.shape[0];
          Genten::Array weights(nc, static_cast<ttb_real *>(w_info.ptr), !copy);

          // Get factors
          const ttb_indx nd = f.size();
          Genten::FacMatArray factors(nd);
          for (ttb_indx i=0; i<nd; ++i) {
            auto mat = py::cast<py::array_t<ttb_real,py::array::c_style> >(f[i]);
            py::buffer_info mat_info = mat.request();
            if (mat_info.ndim != 2)
              throw std::runtime_error("Incompatible buffer dimension!");
            const ttb_indx nrow = mat_info.shape[0];
            if (mat_info.shape[1] != nc)
              throw std::runtime_error("Invalid number of columns!");
            const ttb_indx s0 = mat_info.strides[0]/sizeof(ttb_real);
            const ttb_indx s1 = mat_info.strides[1]/sizeof(ttb_real);
            if (s1 != 1) {
              std::string msg = "Buffer is not layout-right!  Dims = (" +
                std::to_string(nrow) + ", " + std::to_string(nc) +
                "), strides = (" + std::to_string(s0) + ", " +
                std::to_string(s1) + ")";
              throw std::runtime_error(msg);
            }
            ttb_real *ptr = static_cast<ttb_real *>(mat_info.ptr);
            if (s0 == nc) {
              Kokkos::View<ttb_real**, Kokkos::LayoutRight, Genten::DefaultHostExecutionSpace> v(ptr, nrow, nc);
              Genten::FacMatrix A;
              if (copy) {
                A = Genten::FacMatrix(nrow, nc);
                deep_copy(A.view(), v);
              }
              else {
                A = Genten::FacMatrix(nrow, nc, v);
              }
              factors.set_factor(i, A);
            }
            else {
              Kokkos::LayoutStride layout;
              layout.dimension[0] = nrow;
              layout.dimension[1] = nc;
              layout.stride[0] = s0;
              layout.stride[1] = s1;
              Kokkos::View<ttb_real**, Kokkos::LayoutStride, Genten::DefaultHostExecutionSpace> v(ptr, layout);
              // for strided, we always have to copy because FacMatrix must
              // be contiguous
              Genten::FacMatrix A(nrow, nc);
              deep_copy(A.view(), v);
              factors.set_factor(i, A);
            }
          }
          Genten::Ktensor u(weights, factors);
          return u;
        }),"constructor from weights and factor matrices", py::arg("weights"), py::arg("factor_matrices"), py::arg("copy") = true);


    // cl.def("setWeightsRand", (void (Genten::Ktensor::*)()) &Genten::Ktensor::setWeightsRand, "Set all entries to random values between 0 and 1.  Does not change the matrix array, so the Ktensor can become inconsistent");
    // cl.def("setWeights", [](Genten::Ktensor const &o, ttb_real val) -> void { o.setWeights(val); }, "Set all weights equal to val.", py::arg("val"));
    // cl.def("setWeights", [](Genten::Ktensor const &o, const Genten::Array &newWeights) -> void { o.setWeights(newWeights); }, "Set all weights equal to val.", py::arg("newWeights"));
    // cl.def("setMatrices", (void (Genten::Ktensor::*)(ttb_real)) &Genten::Ktensor::setMatrices, "Set all matrix entries equal to val.", py::arg("val"));
    // cl.def("setMatricesRand", (void (Genten::Ktensor::*)()) &Genten::Ktensor::setMatricesRand, "Set all entries to random values in [0,1).");
    // cl.def("setMatricesScatter", (void (Genten::Ktensor::*)(const bool, const bool, Genten::RandomMT &)) &Genten::Ktensor::setMatricesScatter, "Set all entries to reproducible random values.", py::arg("bUseMatlabRNG"), py::arg("bUseParallelRNG"), py::arg("cRMT"));
    // cl.def("setRandomUniform", (void (Genten::Ktensor::*)(const bool, Genten::RandomMT &)) &Genten::Ktensor::setRandomUniform, "Fill the Ktensor with uniform random values, normalized to be stochastic.", py::arg("bUseMatlabRNG"), py::arg("cRMT"));
    // cl.def("scaleRandomElements", (void (Genten::Ktensor::*)()) &Genten::Ktensor::scaleRandomElements, "multiply (plump) a fraction (indices randomly chosen) of each FacMatrix by scale.");
    //setProcessorMap - ProcessorMap
    // cl.def("getProcessorMap", (const Genten::ProcessorMap* (Genten::Ktensor::*)()) &Genten::Ktensor::getProcessorMap, "Get parallel processor map", py::return_value_policy::reference);
    // cl.def("ncomponents", (ttb_indx (Genten::Ktensor::*)()) &Genten::Ktensor::ncomponents, "Return number of components.");
    // cl.def("ndims", (ttb_indx (Genten::Ktensor::*)()) &Genten::Ktensor::ndims, "Return number of dimensions of Ktensor.");
    // cl.def("isConsistent", [](Genten::Ktensor const &o) -> bool { return o.isConsistent(); }, "Consistency check on sizes.");
    // cl.def("isConsistent", [](Genten::Ktensor const &o, const Genten::IndxArray & sz) -> bool { return o.isConsistent(sz); }, "Consistency check on sizes.");
    // cl.def("hasNonFinite", (bool (Genten::Ktensor::*)(ttb_indx &)) &Genten::Ktensor::hasNonFinite, "", py::arg("bad"));
    // cl.def("isNonnegative", (bool (Genten::Ktensor::*)(bool)) &Genten::Ktensor::isNonnegative, "", py::arg("bDisplayErrors"));
    // cl.def("weights", [](Genten::Ktensor const &o) -> Genten::Array { return o.weights(); }, "Return reference to weights vector.");
    // cl.def("weights", [](Genten::Ktensor const &o, ttb_indx i) -> ttb_real { return o.weights(i); }, "Return reference to weights vector.", py::arg("i"));
    cl.def("__getitem__", [](Genten::Ktensor const &o, ttb_indx n) -> const Genten::FacMatrix & { return o[n]; }, "Return a reference to the n-th factor matrix", py::arg("n"));
    cl.def("__str__", [](const Genten::Ktensor& u) {
        std::stringstream ss;
        Genten::print_ktensor(u, ss);
        return ss.str();
      });
    cl.def_property_readonly("pmap", &Genten::Ktensor::getProcessorMap, py::return_value_policy::reference);
    cl.def_property_readonly("ndims", &Genten::Ktensor::ndims);
    cl.def_property_readonly("ncomponents", &Genten::Ktensor::ncomponents);
    cl.def_property_readonly("shape", [](const Genten::Ktensor& u) {
        const ttb_indx nd = u.ndims();
        auto sz = py::tuple(nd);
        for (ttb_indx i=0; i<nd; ++i)
          sz[i] = u[i].nRows();
        return sz;
      });
    cl.def_property("weights",[](const Genten::Ktensor& u) {
        const ttb_indx nc = u.ncomponents();
        // From https://github.com/pybind/pybind11/issues/1042, use a
        // py::capsule to (1) make w a view of u.weights instead of a copy and
        // (2) ensure the memory w wraps does not get deleted out from
        // underneath it by incrementing the reference count now and
        // decrementing it when w is destroyed in python
        Genten::Array weights = u.weights();
        py::capsule capsule(new Genten::Array(weights), [](void *v) { delete reinterpret_cast<Genten::Array*>(v); });
        auto w = py::array_t<ttb_real>({nc}, {sizeof(ttb_real)}, weights.ptr(), capsule);
        return w;
      }, [](const Genten::Ktensor& u, const py::array_t<ttb_real>& w) {
        py::buffer_info w_info = w.request();
        if (w_info.ndim != 1)
          throw std::runtime_error("Incompatible buffer dimension!");
        const ttb_indx nc = u.ncomponents();
        if (nc != w_info.shape[0])
          throw std::runtime_error("Incompatible buffer length!");
        Genten::Array weights(nc, static_cast<ttb_real *>(w_info.ptr), true);
        deep_copy(u.weights(), weights);
      });
  }
}

void pygenten_tensor(py::module &m){
  {
    py::class_<Genten::Tensor, std::shared_ptr<Genten::Tensor>> cl(m, "Tensor");
    cl.def( py::init( [](){ return new Genten::Tensor(); } ), "Empty constructor" );
    cl.def( py::init( [](const Genten::Tensor& src){ return new Genten::Tensor(src); } ), "Copy constructor", py::arg("src") );
    cl.def( py::init( [](const Genten::IndxArray &sz){ return new Genten::Tensor(sz); } ), "Construct tensor of given size initialized to val", py::arg("sz"));
    cl.def( py::init( [](const Genten::IndxArray &sz, ttb_real val){ return new Genten::Tensor(sz, val); } ), "Construct tensor of given size initialized to val", py::arg("sz"), py::arg("val"));
    cl.def( py::init( [](const Genten::IndxArray &sz, const Genten::Array &vals){ return new Genten::Tensor(sz, vals); } ), "Construct tensor with given size and values", py::arg("sz"), py::arg("vals"));

    cl.def(py::init([](const py::array_t<ttb_real,py::array::c_style>& b) {
          // Initialize a Genten::Tensor from a numpy array using "C" layout,
          // which requires a transpose
          py::buffer_info info = b.request();

          // Tensor size
          const ttb_indx nd = info.ndim;
          Genten::IndxArray sz(nd), szt(nd);
          ttb_indx numel = 1;
          for (ttb_indx i=0; i<nd; ++i) {
            sz[i]  = info.shape[i];      // size of this tensor
            szt[i] = info.shape[nd-i-1]; // size of transposed tensor
            numel *= sz[i];
          }

          // Tensor values
          // To do:  make this parallel.  But we should combine that with
          // putting the tensor on the device.
          Genten::Array vals(numel);
          Genten::IndxArray s(nd), st(nd);
          ttb_real *ptr = static_cast<ttb_real*>(info.ptr);
          for (ttb_indx i=0; i<numel; ++i) {

            // Map linearized index to mult-index
            Genten::Impl::ind2sub(s, sz, numel, i);

            // Compute multi-index of transposed tensor
            for (ttb_indx j=0; j<nd; ++j)
              st[j] = s[nd-j-1];

            // Map transpose multi-index to transposed linearized index
            const ttb_indx k = Genten::Impl::sub2ind(st, szt);

            // Copy corresponding values
            vals[i] = ptr[k];
          }
          Genten::Tensor x(sz, vals);
          return x;
        }));
    cl.def(py::init([](const py::array_t<ttb_real,py::array::f_style>& b) {
          // Initialize a Genten::Tensor from a numpy array using "F" layout,
          // which is the same as GenTen's layout, so no need to transpose
          py::buffer_info info = b.request();

          // Tensor size
          const ttb_indx nd = info.ndim;
          Genten::IndxArray sz(nd);
          ttb_indx numel = 1;
          for (ttb_indx i=0; i<nd; ++i) {
            sz[i]  = info.shape[i];      // size of this tensor
            numel *= sz[i];
          }

          // Tensor values.  We use a view to avoid the copy.  This should be
          // safe because I believe we aren't alloying pybind to create a
          // temporary because we did not include py::array::forcecast in the
          // template parameters.
          Genten::Array vals(numel, static_cast<ttb_real*>(info.ptr), true);
          Genten::Tensor x(sz, vals);
          return x;
        }));

    cl.def("ndims", (ttb_indx (Genten::Tensor::*)()) &Genten::Tensor::ndims, "Return the number of dimensions (i.e., the order).");
    cl.def("size", [](Genten::Tensor const &o, ttb_indx i) -> ttb_indx { return o.size(i); } , "Return size of dimension i.", py::arg("i"));
    cl.def("size", [](Genten::Tensor const &o) -> Genten::IndxArray { return o.size(); } , "Return sizes array.");
    cl.def("numel", (ttb_indx (Genten::Tensor::*)()) &Genten::Tensor::numel, "Return the total number of elements in the tensor.");
    cl.def("values", [](Genten::Tensor const &o) -> Genten::Array { return o.getValues(); } , "Return data array.");
    cl.def("__str__", [](const Genten::Tensor& X) {
        std::stringstream ss;
        Genten::print_tensor(X, ss);
        return ss.str();
      });
    cl.def("getProcessorMap", (const Genten::ProcessorMap* (Genten::Tensor::*)()) &Genten::Tensor::getProcessorMap, "Get parallel processor map", py::return_value_policy::reference);
  }
}

void pygenten_sptensor(py::module &m){
  {
    py::class_<Genten::Sptensor, std::shared_ptr<Genten::Sptensor>> cl(m, "Sptensor");
    cl.def( py::init( [](){ return new Genten::Sptensor(); } ), "Empty constructor" );
    cl.def( py::init( [](const Genten::Sptensor& src){ return new Genten::Sptensor(src); } ), "Copy constructor", py::arg("src") );
    cl.def( py::init( [](const Genten::IndxArray &sz, ttb_indx nz){ return new Genten::Sptensor(sz,nz); } ), "Constructor for a given size and number of nonzeros", py::arg("sz"), py::arg("nz"));

    cl.def(py::init([](const py::tuple& sizes, const py::array_t<std::int64_t,py::array::c_style>& subs, const py::array_t<ttb_real,py::array::c_style>& vals) {
          // Sizes
          const ttb_indx nd = sizes.size();
          Genten::IndxArray sz(nd);
          for (ttb_indx i=0; i<nd; ++i)
            sz[i] = py::cast<ttb_indx>(sizes[i]);

          // Subscripts
          py::buffer_info subs_info = subs.request();
          if (subs_info.ndim != 2)
            throw std::runtime_error("Incompatible subs dimension!");
          if (subs_info.shape[1] != nd)
            throw std::runtime_error("Invalid number of subscript columns!");
          const ttb_indx nnz = subs_info.shape[0];
          std::int64_t *subs_ptr = static_cast<std::int64_t *>(subs_info.ptr);
          typename Genten::Sptensor::subs_view_type s("subs", nnz, nd);
          for (ttb_indx i=0; i<nnz; ++i)
            for (ttb_indx j=0; j<nd; ++j)
              s(i,j) = subs_ptr[i*nd+j];

          // Values.  TTB stores it as a 2-D array for some reason
          py::buffer_info vals_info = vals.request();
          if (vals_info.ndim != 1 && vals_info.ndim != 2)
            throw std::runtime_error("Incompatible vals dimension!");
          if (vals_info.shape[0] != nnz)
            throw std::runtime_error("Invalid number of value rows!");
          if (vals_info.ndim == 2 && vals_info.shape[1] != 1)
            throw std::runtime_error("Invalid number of value columns!");
          ttb_real *vals_ptr = static_cast<ttb_real *>(vals_info.ptr);
          typename Genten::Sptensor::vals_view_type v(vals_ptr, nnz);

          Genten::Sptensor x(sz, v, s);
          return x;
        }));

    cl.def("ndims", (ttb_indx (Genten::Sptensor::*)()) &Genten::Sptensor::ndims, "Return the number of dimensions (i.e., the order).");
    cl.def("size", [](Genten::Sptensor const &o, ttb_indx i) -> ttb_indx { return o.size(i); } , "Return size of dimension i.", py::arg("i"));
    cl.def("size", [](Genten::Sptensor const &o) -> Genten::IndxArray { return o.size(); } , "Return sizes array.");
    cl.def("numel", (ttb_real (Genten::Sptensor::*)()) &Genten::Sptensor::numel, "Return the total number of elements in the tensor.");
    cl.def("nnz", (ttb_indx (Genten::Sptensor::*)()) &Genten::Sptensor::nnz, "Return the total number of nonzeros.");
    cl.def("__str__", [](const Genten::Sptensor& X) {
        std::stringstream ss;
        Genten::print_sptensor(X, ss);
        return ss.str();
      });
    cl.def("getProcessorMap", (const Genten::ProcessorMap* (Genten::Sptensor::*)()) &Genten::Sptensor::getProcessorMap, "Get parallel processor map", py::return_value_policy::reference);
  }
  {
    py::class_<Genten::DTC, std::shared_ptr<Genten::DTC>> cl(m, "DistTensorContext");
    cl.def( py::init( [](){ return new Genten::DTC(); } ), "Empty constructor" );
    cl.def("getProcessorMap", [](const Genten::DTC& dtc) {
        return dtc.pmap_ptr();
      }, "Get parallel processor map");
    cl.def("distributeTensor",[](Genten::DTC& dtc, const std::string& file, const ttb_indx index_base, const bool compressed, const std::string& json, const Genten::AlgParams& algParams) {
        py::scoped_ostream_redirect stream(
          std::cout,                                // std::ostream&
          py::module_::import("sys").attr("stdout") // Python output
          );
        py::scoped_ostream_redirect err_stream(
          std::cerr,                                // std::ostream&
          py::module_::import("sys").attr("stderr") // Python output
          );
        Genten::ptree tree;
        tree.parse(json);
        Genten::Sptensor X_sparse;
        Genten::Tensor X_dense;
        dtc.distributeTensor(file, index_base, compressed, tree,
                             algParams, X_sparse, X_dense);
        return std::make_tuple(X_sparse,X_dense);
      }, "Read a tensor from a file and distribute it in parallel");
    cl.def("distributeTensor", [](Genten::DTC& dtc, const Genten::Sptensor& X, const Genten::AlgParams& algParams) {
        return dtc.distributeTensor(X, algParams);
      }, "Distribute a given sparse tensor in parallel");
    cl.def("distributeTensor", [](Genten::DTC& dtc, const Genten::Tensor& X, const Genten::AlgParams& algParams) {
        return dtc.distributeTensor(X, algParams);
      }, "Distribute a given dense tensor in parallel");
    cl.def("importToAll", [](const Genten::DTC& dtc, const Genten::Ktensor& u) {
        return dtc.importToAll<Genten::DefaultHostExecutionSpace>(u);
      }, "Import a Ktensor to all processors");
    cl.def("importToRoot", [](const Genten::DTC& dtc, const Genten::Ktensor& u) {
        return dtc.importToRoot<Genten::DefaultHostExecutionSpace>(u);
      }, "Import a Ktensor to root processor");
    cl.def("exportFromRoot", [](const Genten::DTC& dtc, const Genten::Ktensor& u) {
        return dtc.exportFromRoot<Genten::DefaultHostExecutionSpace>(u);
      }, "Export a Ktensor from the root processor to all");
  }
}

void pygenten_classes(py::module &m){
  pygenten_perfhistory(m);
  pygenten_algparams(m);
  pygenten_proc_map(m);
  pygenten_ktensor(m);
  pygenten_tensor(m);
  pygenten_sptensor(m);
}
