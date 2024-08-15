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
    py::class_<Genten::PerfHistory::Entry, std::shared_ptr<Genten::PerfHistory::Entry>> cl(m, "Entry",R"(
    Class storing performance information from one iteration of a tensor
    decomposition method.  Not all methods fill all properties.)");
    cl.def(py::init([](){ return new Genten::PerfHistory::Entry(); }), R"(
     Constructor that returns an empty Entry.)");
    cl.def_readwrite("iteration", &Genten::PerfHistory::Entry::iteration, R"(
     The solver iteration this entry corresponds to.)");
    cl.def_readwrite("residual", &Genten::PerfHistory::Entry::residual, R"(
     The residual of the objective function for this iteration.)");
    cl.def_readwrite("fit", &Genten::PerfHistory::Entry::fit, R"(
     The L2 fit for this iteration, i.e., 1-||X-M||_F/||X||_F.)");
    cl.def_readwrite("grad_norm", &Genten::PerfHistory::Entry::grad_norm, R"(
     Norm of the objective function gradient for this iteration.)");
    cl.def_readwrite("cum_time", &Genten::PerfHistory::Entry::cum_time, R"(
     Cummulative wall-clock time up to and including this iteration.)");
    cl.def_readwrite("mttkrp_throughput", &Genten::PerfHistory::Entry::mttkrp_throughput,R"(
     MTTKRP memory throughput.)");
    cl.def("__str__", [](Genten::PerfHistory::Entry const &o) -> std::string {
        std::ostringstream s;
        s << o.iteration << " " << o.residual << " " << o.fit << " " << o.grad_norm << " " << o.cum_time << " " << o.mttkrp_throughput;
        return s.str();
      }, R"(
     Return a string representation of this entry.)" );
  }
  {
    py::class_<Genten::PerfHistory, std::shared_ptr<Genten::PerfHistory>> cl(m, "PerfHistory",R"(
    Class storing performance history over iterations of a tensor
    decomposition method by storing a list of objects of type Entry.)");
    cl.def(py::init([](){ return new Genten::PerfHistory(); }), R"(
     Constructor that returns an empty PerfHistory.)");
    cl.def("addEntry", (void (Genten::PerfHistory::*)(const struct Genten::PerfHistory::Entry &)) &Genten::PerfHistory::addEntry, R"(
     Add the given entry to the end of the list.)", py::arg("entry"));
    cl.def("addEntry", (void (Genten::PerfHistory::*)()) &Genten::PerfHistory::addEntry, R"(
     Add an empty entry to the list.)");
    cl.def("getEntry", (struct Genten::PerfHistory::Entry & (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::getEntry, R"(
     Get entry i from the list.)", py::arg("i"));
    cl.def("__getitem__", (struct Genten::PerfHistory::Entry & (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::getEntry, R"(
     Get entry i from the list.)", py::arg("i"));
    cl.def("lastEntry", (struct Genten::PerfHistory::Entry & (Genten::PerfHistory::*)()) &Genten::PerfHistory::lastEntry, R"(
     Get the last entry from the list.)");
    cl.def("size", (ttb_indx (Genten::PerfHistory::*)()) &Genten::PerfHistory::size, R"(
     Return the number of entries)");
    cl.def("resize", (void (Genten::PerfHistory::*)(const ttb_indx)) &Genten::PerfHistory::resize, R"(
     Resize the list to the given new size)", py::arg("new_size"));
    cl.def("__str__", [](Genten::PerfHistory const &o) -> std::string {
        std::ostringstream s;
        o.print(s);
        return s.str();
      }, R"(
     Return a string representation of all entries.)");
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
    .value("GCP_FED", Genten::Solver_Method::type::GCP_FED)
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
    .value("MttkrpRowBased", Genten::MTTKRP_Method::type::RowBased)
    .value("MttkrpPhan", Genten::MTTKRP_Method::type::Phan)
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
    .value("Tpetra", Genten::Dist_Update_Method::type::Tpetra)
    .value("AllGatherReduce", Genten::Dist_Update_Method::type::AllGatherReduce)
    .value("OneSided", Genten::Dist_Update_Method::type::OneSided)
    .value("TwoSided", Genten::Dist_Update_Method::type::TwoSided)
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
  py::enum_<Genten::GCP_FedMethod::type>(m, "GCP_FedMethod")
    .value("FedOpt", Genten::GCP_FedMethod::type::FedOpt)
    .value("FedAvg", Genten::GCP_FedMethod::type::FedAvg)
    .export_values();
  py::enum_<Genten::GCP_AnnealerMethod::type>(m, "GCP_AnnealerMethod")
    .value("Traditional", Genten::GCP_AnnealerMethod::type::Traditional)
    .value("Cosine", Genten::GCP_AnnealerMethod::type::Cosine)
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
    py::class_<Genten::AlgParams, std::shared_ptr<Genten::AlgParams>> cl(m, "AlgParams", R"(
    Class for pass algorithm parameters to GenTen solvers.

    This class is not intended to be used directly by users.  Instead users
    should pass arguments as a JSON structure or a dict.)");
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
    cl.def_readwrite("ftol", &Genten::AlgParams::ftol);
    cl.def_readwrite("gtol", &Genten::AlgParams::gtol);
    cl.def_readwrite("memory", &Genten::AlgParams::memory);
    cl.def_readwrite("sub_iters", &Genten::AlgParams::sub_iters);
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
    cl.def_readwrite("annealer", &Genten::AlgParams::annealer);
    cl.def_readwrite("anneal_min_lr", &Genten::AlgParams::anneal_min_lr);
    cl.def_readwrite("anneal_max_lr", &Genten::AlgParams::anneal_max_lr);
    cl.def_readwrite("anneal_temp", &Genten::AlgParams::anneal_Ti);

    cl.def_readwrite("fed_method", &Genten::AlgParams::fed_method);
    cl.def_readwrite("meta_step_type", &Genten::AlgParams::meta_step_type);
    cl.def_readwrite("meta_rate", &Genten::AlgParams::meta_rate);

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
#ifdef HAVE_PYTHON_EMBED
    cl.def("set_py_goal", [](Genten::AlgParams& a, const py::object& po) {
        a.python_object = po;
        a.goal_method = Genten::GCP_Goal_Method::PythonObject;
      });
#endif
    cl.def("parse_json", [](Genten::AlgParams& a, const std::string& str) {
        Genten::ptree tree;
        tree.parse(str);
        a.parse(tree);
      });
  }
}

void pygenten_proc_map(py::module &m){
  {
    py::class_<Genten::ProcessorMap, std::shared_ptr<Genten::ProcessorMap>> cl(m, "ProcessorMap", R"(
    Class for managing MPI information related to distributed  parallel tensor
    decompositions.)

    GenTen uses a Cartesian grid decomposition approach where an N-way tensor
    is partitioned along an N-way grid of MPI ranks.  Subcommunicators are
    constructed along each dimension of the grid.)");
    cl.def("gridSize", (ttb_indx (Genten::ProcessorMap::*)()) &Genten::ProcessorMap::gridSize, R"(
    Return total number of processors in the grid)");
    cl.def("gridRank", (ttb_indx (Genten::ProcessorMap::*)()) &Genten::ProcessorMap::gridRank, R"(
    Return rank of this processor in the grid.)");
    cl.def("gridAllReduceArray", [](const Genten::ProcessorMap& pmap, py::buffer b, Genten::ProcessorMap::MpiOp op = Genten::ProcessorMap::Sum) {
        py::buffer_info info = b.request();
        if (info.format != py::format_descriptor<ttb_real>::format())
          throw std::runtime_error("Incompatible format: expected a ttb_real array!");
        if (info.ndim != 1)
          throw std::runtime_error("Incompatible buffer dimension!");
        ttb_real *ptr = static_cast<ttb_real*>(info.ptr);
        ttb_indx n = info.shape[0];
        pmap.gridAllReduce(ptr, n, op);
      }, R"(
    Parallel reduce the given array across all processors in the grid.

    The given array 'b' can by any 1-D array type that supports the buffer
    protocol, such as a numpy.nparray.  The operation for combining elements
    is given by 'op', which defaults to Sum.  The array 'b' is overwritten with
    the result on each processor.)", py::arg("b"), py::arg("op") = Genten::ProcessorMap::Sum);
    cl.def("gridAllReduce", [](const Genten::ProcessorMap& pmap, ttb_real x, Genten::ProcessorMap::MpiOp op) {
        return pmap.gridAllReduce(x, op);
      }, R"(
    Parallel reduce the given scalar across all processors in the grid and
    return the result.

    The operation for combining elements across processors is given by 'op',
    which defaults to Sum.)", py::arg("x"), py::arg("op") = Genten::ProcessorMap::Sum);
  }
}

void pygenten_ktensor(py::module &m){
  {
    py::class_<Genten::Ktensor, std::shared_ptr<Genten::Ktensor>> cl(m, "Ktensor",R"(
    Class for Kruskal tensors in GenTen mirroring pyttb::ktensor.

    Contains the following data members:
      * weights: 1-D numpy.ndarray containing the weights of the rank-1
        tensors defined by the outer products of the column vectors of the
        factor_matrices (read/write).
      * factor_matrices: tuple of 2-D numpy.ndarray. The length of the tuple is
        equal to the number of dimensions of the tensor. The shape of the ith
        element of the list is (n_i, r), where n_i is the length dimension i
        and r is the rank of the tensor (as well as the length of the weights
        vector) (read-only).)
     Note that while factor_matrices, as a tuple is read-only, the
     individual factor matrices within the tuple can be written to using slice
     syntax.)");
    cl.def(py::init([](){ return new Genten::Ktensor(); }),R"(
     Constructor that returns an empty Ktensor.)");
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
            auto mat = py::cast<py::buffer>(f[i]);
            py::buffer_info mat_info = mat.request();
            if (mat_info.format != py::format_descriptor<ttb_real>::format())
                throw std::runtime_error("Incompatible format: expected a ttb_real array!");
            if (mat_info.ndim != 2)
              throw std::runtime_error("Incompatible buffer dimension!");
            const ttb_indx nrow = mat_info.shape[0];
            if (static_cast<ttb_indx>(mat_info.shape[1]) != nc)
              throw std::runtime_error("Invalid number of columns!");
            const ttb_indx s0 = mat_info.strides[0]/sizeof(ttb_real);
            const ttb_indx s1 = mat_info.strides[1]/sizeof(ttb_real);
            ttb_real *ptr = static_cast<ttb_real *>(mat_info.ptr);
            if (s0 == nc && s1 == 1) { // LayoutRight case
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
            else if (s0 == 1 && s1 == nrow) { // LayoutLeft case
              Kokkos::View<ttb_real**, Kokkos::LayoutLeft, Genten::DefaultHostExecutionSpace> v(ptr, nrow, nc);
              Genten::FacMatrix A(nrow, nc);
              deep_copy(A.view(), v);
              factors.set_factor(i, A);
            }
            else { // General case
              Kokkos::LayoutStride layout;
              layout.dimension[0] = nrow;
              layout.dimension[1] = nc;
              layout.stride[0] = s0;
              layout.stride[1] = s1;
              Kokkos::View<ttb_real**, Kokkos::LayoutStride, Genten::DefaultHostExecutionSpace> v(ptr, layout);
              Genten::FacMatrix A(nrow, nc);
              deep_copy(A.view(), v);
              factors.set_factor(i, A);
            }
          }
          Genten::Ktensor u(weights, factors);

          // Tie w and f to u to prevent python from deleting them
          // Because each matrix could have a different layout, we set the
          // extra data if a view was requested, regardless of which matrices
          // were/weren't copied
          if (!copy)
            u.set_extra_data(std::make_pair(w, f));

          return u;
        }),R"(
    Constructor from given weights and factor matrices.

    Parameters:
      * weights: 1-D numpy.ndarray of weights of length r, where r is the rank
        of the Ktensor.
      * factor_matrices: list of 2-D numpy.ndarray matrices of length n where
        n is the number of dimensions.  Each factor matrix must have r columns.
      * copy: bool
          Whether to copy weights and factor_matrices or alias the given ones.)", py::arg("weights"), py::arg("factor_matrices"), py::arg("copy") = true);

    cl.def("__getitem__", [](const Genten::Ktensor&u, ttb_indx dim) {
        const auto mat = u[dim];
        const ttb_indx m = mat.nRows();
        const ttb_indx n = mat.nCols();
        const ttb_indx s0 = mat.view().stride(0)*sizeof(ttb_real);
        const ttb_indx s1 = mat.view().stride(1)*sizeof(ttb_real);
        py::capsule capsule(new Genten::FacMatrix(mat), [](void *v) { delete reinterpret_cast<Genten::FacMatrix*>(v); });
        auto fac_mat = py::array_t<ttb_real>({m, n}, {s0, s1}, mat.view().data(), capsule);
        return fac_mat;
      }, R"(
    Return the n-th factor matrix as a numpy.ndarray.

    The returned numpy.ndarray aliases the internal factor matrices and thus may
    be used to change their values.)", py::arg("n"));
    cl.def("full", [](const Genten::Ktensor& u){ return new Genten::Tensor(u); },R"(
     Constructor a dense tensor from this Ktensor u.

     The Ktensor is multiplied out to reconstruct the full, dense tensor.)");
    cl.def_property_readonly("is_values_view", [](const Genten::Ktensor& u) {
        return u.has_extra_data();
      }, R"(
    Return whether this Ktensor is a view of numpy arrays for weights and
    factor matrices (i.e., constructed with copy=False).)");
    cl.def("__str__", [](const Genten::Ktensor& u) {
        std::stringstream ss;
        Genten::print_ktensor(u, ss);
        return ss.str();
      }, R"(
    Returns a string representation of the Ktensor.)");
    cl.def_property_readonly("pmap", &Genten::Ktensor::getProcessorMap, py::return_value_policy::reference, R"(
    Processor map for distributed memory parallelism.)");
    cl.def_property_readonly("ndims", &Genten::Ktensor::ndims, R"(
    Number of dimensions of the Ktensor as determined by the length of
    the factor_matrices property.)");
    cl.def_property_readonly("ncomponents", &Genten::Ktensor::ncomponents, R"(
    The rank of the Ktensor as determined by the length of weights and the
    number of columns of each factor matrix.)");
    cl.def_property_readonly("shape", [](const Genten::Ktensor& u) {
        const ttb_indx nd = u.ndims();
        auto sz = py::tuple(nd);
        for (ttb_indx i=0; i<nd; ++i)
          sz[i] = u[i].nRows();
        return sz;
      }, R"(
    The dimensions of the Ktensor as determined by the number of rows of each
    factor matrix.)");
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
        if (nc != static_cast<ttb_indx>(w_info.shape[0]))
          throw std::runtime_error("Incompatible buffer length!");
        Genten::Array weights(nc, static_cast<ttb_real *>(w_info.ptr), true);
        deep_copy(u.weights(), weights);
      }, R"(
    The weights array as a (read-write) numpy.ndarray.)");
    cl.def_property_readonly("factor_matrices",[](const Genten::Ktensor& u) {
        const ttb_indx nd = u.ndims();
        // Return u.factor_matrices as a tuple instead of a list to make it
        // immutable because while we alias each factor matrix, we don't (and
        // can't) alias the list itself.  By making it immutable, we prevent
        // assignments like "u.factor_matrices[0] = np.zeros(...)", which won't
        // work as expected (you can still write to the factor matrices using
        // slices though, e.g., "u.factor_matrices[0][:,:] = np.zeros(...)",
        // which works as expected).
        py::tuple fac_mats(nd);
        for (ttb_indx dim=0; dim<nd; ++dim) {
          const auto mat = u[dim];
          const ttb_indx m = mat.nRows();
          const ttb_indx n = mat.nCols();
          const ttb_indx s0 = mat.view().stride(0)*sizeof(ttb_real);
          const ttb_indx s1 = mat.view().stride(1)*sizeof(ttb_real);
          py::capsule capsule(new Genten::FacMatrix(mat), [](void *v) { delete reinterpret_cast<Genten::FacMatrix*>(v); });
          auto fac_mat = py::array_t<ttb_real>({m, n}, {s0, s1}, mat.view().data(), capsule);
          fac_mats[dim] = fac_mat;
        }
        return fac_mats;
      }, R"(
    The factor matrices returned as a (read-only) tuple of numpy.ndarray.

    While the tuple is read-only, entries in each factor matrix may be changed
    using index/slice syntax.)");
  }
}

void pygenten_tensor(py::module &m){
  {
    py::class_<Genten::Tensor, std::shared_ptr<Genten::Tensor>> cl(m, "Tensor",R"(
    Class for dense tensors in GenTen mirroring pyttb::tensor.

    Contains the following data members:
      * data: n-D numpy.ndarray where n is the dimension of the tensor.)");
    cl.def(py::init([](){ return new Genten::Tensor(); }),R"(
     Constructor that returns an empty tensor.)");
    cl.def(py::init([](const py::buffer& b, const bool copy=true) {
        // Initialize a Genten::Tensor from a numpy array
        py::buffer_info info = b.request();

        if (info.format != py::format_descriptor<ttb_real>::format())
          throw std::runtime_error("Incompatible format: expected a ttb_real array!");

        // Tensor size
        const ttb_indx nd = info.ndim;
        Genten::IndxArray sz(nd);
        ttb_indx numel = 1;
        for (ttb_indx i=0; i<nd; ++i) {
          sz[i]  = info.shape[i];      // size of this tensor
          numel *= sz[i];
        }

        Genten::Tensor x;

        // Check for empty tensor (nd may be > 0 in this case)
        if (numel == 0)
          return x;

        if (info.strides[0]/sizeof(ttb_real) == 1) {
          // The 'F' layout case
          Genten::Array vals(numel, static_cast<ttb_real*>(info.ptr), !copy);
          x = Genten::Tensor(sz, vals, Genten::TensorLayout::Left);

          if (!copy)
            x.set_extra_data(b);
        }
        else if (info.strides[nd-1]/sizeof(ttb_real) == 1) {
          // The 'C' layout case
          Genten::Array vals(numel, static_cast<ttb_real*>(info.ptr), !copy);
          x = Genten::Tensor(sz, vals, Genten::TensorLayout::Right);

          if (!copy)
            x.set_extra_data(b);
        }
        else
          throw std::runtime_error("Incompatible array layout.  Must be 'C' or 'F' layout!");
        return x;
      }),R"(
    Constructor from the given buffer such as a numpy.ndarray.

    Both 'F' and 'C' orderings are supported, but not a general strided ordering.)", py::arg("b"), py::arg("copy") = true);
    cl.def(py::init([](const Genten::Ktensor& u){ return new Genten::Tensor(u); }),R"(
     Constructor that creates a dense tensor from the supplied Ktensor u.

     The Ktensor is multiplied out to reconstruct the full, dense tensor.)", py::arg("u"));
    cl.def(py::init([](const Genten::Sptensor& X){ return new Genten::Tensor(X); }),R"(
     Constructor that creates a dense tensor from the supplied sparse tensor X.)", py::arg("X"));

    cl.def_property_readonly("pmap", &Genten::Tensor::getProcessorMap, py::return_value_policy::reference, R"(
    Processor map for distributed memory parallelism.)");
    cl.def_property_readonly("ndims", &Genten::Tensor::ndims, R"(
    Number of dimensions of the tensor.)");
    cl.def_property_readonly("shape", [](const Genten::Tensor& X) {
        const ttb_indx nd = X.ndims();
        auto sz = py::tuple(nd);
        for (ttb_indx i=0; i<nd; ++i)
          sz[i] = X.size(i);
        return sz;
      }, R"(
    The dimensions of the tensor.)");
    cl.def_property_readonly("data",[](const Genten::Tensor& X) {
        Genten::Array vals = X.getValues();
        py::capsule capsule(new Genten::Array(vals), [](void *v) { delete reinterpret_cast<Genten::Array*>(v); });
        const ttb_indx nd = X.ndims();
        std::vector<ttb_indx> shape(nd), strides(nd);
        for (ttb_indx i=0; i<nd; ++i)
          shape[i] = X.size(i);
        if (X.has_left_impl()) {
          strides[0] = sizeof(ttb_real);
          for (ttb_indx i=1; i<nd; ++i)
            strides[i] = strides[i-1]*shape[i-1];
        }
        else {
          strides[nd-1] = sizeof(ttb_real);
          for (ttb_indx i=nd-1; i>0; --i)
            strides[i-1] = strides[i]*shape[i];
        }
        return py::array_t<ttb_real>(shape, strides, vals.ptr(), capsule);
      }, R"(
    The tensor data returned as a numpy.ndarray.

    The property is read-only, meaning the ndarray cannot be changed, but values
    may be changed using index/slice syntax.)");
    cl.def_property_readonly("nnz", [](const Genten::Tensor& X) {
        return X.nnz();
      }, R"(
    Returns the total number of elements in the tensor.)");
    cl.def_property_readonly("is_values_view", [](const Genten::Tensor& X) {
        return X.has_extra_data();
      }, R"(
    Return whether this tensor is a view of a numpy array (i.e., constructed
    with copy=False).)");
    cl.def_property_readonly("layout", [](const Genten::Tensor& X) {
        if (X.has_left_impl())
          return std::string("F");
        return std::string("C");
      }, R"(
    Return memory layout of the tensor ('F' or 'C').)");
    cl.def("__str__", [](const Genten::Tensor& X) {
        std::stringstream ss;
        Genten::print_tensor(X, ss);
        return ss.str();
      }, R"(
    Returns a string representation of the tensor.)");
  }
}

void pygenten_sptensor(py::module &m){
  {
    py::class_<Genten::Sptensor, std::shared_ptr<Genten::Sptensor>> cl(m, "Sptensor", "Tensor",R"(
    Class for sparse tensors in GenTen mirroring pyttb::sptensor.

    Contains the following data members:
      * vals: 1-D numpy.ndarray of length nnz where nnz is the number of
        nonzeros in the tensor.
      * subs: 2-D numpy.ndarray of signed 64-bit integers with nnz rows and n
        columns, where n is the number of dimensions.  Each row contains the
        coordinates of the corresponding nonzero.
)");
    cl.def(py::init([](){ return new Genten::Sptensor(); }),R"(
     Constructor that returns an empty sptensor.)");
    cl.def(py::init([](const py::tuple& sizes, const py::buffer& subs, const py::buffer& vals, const bool copy=true) {
          // Sizes
          // Check for empty tensor (nd may be > 0 in this case)
          const ttb_indx nd = sizes.size();
          if (nd == 0)
            return Genten::Sptensor();
          Genten::IndxArray sz(nd);
          for (ttb_indx i=0; i<nd; ++i) {
            sz[i] = py::cast<ttb_indx>(sizes[i]);
            if (sz[i] == 0)
              return Genten::Sptensor();
          }

          // Subscripts.  We support nearly any type of ordinal, but only
          // if it matches ttb_indx can we avoid the copy.
          py::buffer_info subs_info = subs.request();
          if (subs_info.ndim != 2)
            throw std::runtime_error("Incompatible subs dimension!");
          if (static_cast<ttb_indx>(subs_info.shape[1]) != nd)
            throw std::runtime_error("Invalid number of subscript columns!");
          const ttb_indx nnz = subs_info.shape[0];
          typename Genten::Sptensor::subs_view_type s;
          auto copy_subs = [=](const auto* subs_ptr) {
            typename Genten::Sptensor::subs_view_type s("subs", nnz, nd);
            for (ttb_indx i=0; i<nnz; ++i)
              for (ttb_indx j=0; j<nd; ++j)
                s(i,j) = subs_ptr[i*nd+j];
            return s;
          };

          // Note: format_descriptor<> does not include long/unsigned long
          // for some reason, so we need to check for them separately.
          // Base on https://python.readthedocs.io/en/v2.7.2/library/struct.html#format-characters, and the format codes in pybind11/common.h,
          // int/unsigned/long long/unsigned long long should map to the same
          // format codes as the std::[u]int??_t types.  We assume ttb_indx
          // is always unsigned though.
          bool copied_subs = true;
          if (subs_info.format == py::format_descriptor<ttb_indx>::format() ||
             (subs_info.format == "L" && sizeof(unsigned long) == sizeof(ttb_indx))) {
            if (subs_info.strides[1] != sizeof(ttb_indx))
              throw std::runtime_error("Subscript array must have 'C' layout!=");
            ttb_indx* subs_ptr = static_cast<ttb_indx *>(subs_info.ptr);
            if (copy) {
              s = typename Genten::Sptensor::subs_view_type("subs", nnz, nd);
              typename Genten::Sptensor::subs_view_type s2(subs_ptr, nnz, nd);
              deep_copy(s, s2);
            }
            else {
              s = typename Genten::Sptensor::subs_view_type(subs_ptr, nnz, nd);
              copied_subs = false;
            }
          }
          else if (subs_info.format == py::format_descriptor<std::uint64_t>::format()) {
            if (subs_info.strides[1] != sizeof(std::uint64_t))
              throw std::runtime_error("Subscript array must have 'C' layout!=");
            s = copy_subs(static_cast<std::int64_t *>(subs_info.ptr));
          }
          else if (subs_info.format == py::format_descriptor<std::int64_t>::format()) {
            if (subs_info.strides[1] != sizeof(std::int64_t))
              throw std::runtime_error("Subscript array must have 'C' layout!=");
            s = copy_subs(static_cast<std::int64_t *>(subs_info.ptr));
          }
          else if (subs_info.format == py::format_descriptor<std::uint32_t>::format()) {
            if (subs_info.strides[1] != sizeof(std::uint32_t))
              throw std::runtime_error("Subscript array must have 'C' layout!=");
            s = copy_subs(static_cast<std::uint32_t *>(subs_info.ptr));
          }
          else if (subs_info.format == py::format_descriptor<std::int32_t>::format()) {
            if (subs_info.strides[1] != sizeof(std::int32_t))
              throw std::runtime_error("Subscript array must have 'C' layout!=");
            s = copy_subs(static_cast<std::int32_t *>(subs_info.ptr));
          }
          else if (subs_info.format == "l") {
            if (subs_info.strides[1] != sizeof(long))
              throw std::runtime_error("Subscript array must have 'C' layout!=");
            s = copy_subs(static_cast<long *>(subs_info.ptr));
          }
          else
            throw std::runtime_error("Incompatible subscript format: expected a int32/int64 or uint32/uint64 array!  Format code is: " + std::string(subs_info.format));

          // Values.  TTB stores it as a 2-D array for some reason
          py::buffer_info vals_info = vals.request();
          if (vals_info.format != py::format_descriptor<ttb_real>::format())
            throw std::runtime_error("Incompatible value format:  expected a ttb_real array!");
          if (vals_info.ndim != 1 && vals_info.ndim != 2)
            throw std::runtime_error("Incompatible vals dimension!");
          if (static_cast<ttb_indx>(vals_info.shape[0]) != nnz)
            throw std::runtime_error("Invalid number of value rows!");
          if (vals_info.ndim == 2 && vals_info.shape[1] != 1)
            throw std::runtime_error("Invalid number of value columns!");
          ttb_real *vals_ptr = static_cast<ttb_real *>(vals_info.ptr);
          typename Genten::Sptensor::vals_view_type v(vals_ptr, nnz);

          Genten::Sptensor x;
          bool copied_vals = true;
          if (copy) {
            typename Genten::Sptensor::vals_view_type v2("vals", nnz);
            deep_copy(v2,v);
            x = Genten::Sptensor(sz, v2, s);
          }
          else {
            x = Genten::Sptensor(sz, v, s);
            copied_vals = false;
          }

          // Tie original subs/vals to x to prevent python from deleteing them
          if (!copied_subs && !copied_vals)
            x.set_extra_data(std::make_pair(subs, vals));
          else if (!copied_subs)
            x.set_extra_data(subs);
          else if (!copied_vals)
            x.set_extra_data(vals);

          return x;
        }),R"(
    Constructor from shape, subscripts, and value arrays.

    Parameters:
      * sizes: tuple containing the dimension of each mode of the tensor.
      * subs: 2-D numpy.ndarray of signed or unsigned 32-bit or 64-bit integers
        containing coordinates of each nonzero.
      * vals: 1-D numpy.ndarray containing values of each nonzero.)", py::arg("sizes"), py::arg("subs"), py::arg("vals"), py::arg("copy") = true);
    cl.def(py::init([](const Genten::Ktensor& u){ return new Genten::Sptensor(Genten::Tensor(u)); }),R"(
     Constructor that creates a sparse tensor from the supplied Ktensor u.

     The Ktensor is multiplied out to reconstruct a full, dense tensor, which
     is then scanned for zeros when producing the sparse tensor.)", py::arg("u"));
    cl.def(py::init([](const Genten::Tensor& X, const ttb_real tol=0.0){ return new Genten::Sptensor(X,tol); }),R"(
     Constructor that creates a sparse tensor from the supplied dense tensor X.

     The dense tensor is scanned for zeros which are excluded from the resulting
     sparse tensor.  Values are considered to be non-zero if they are larger in
     magnitude than the supplied tolerance.)", py::arg("X"), py::arg("tol") = 0.0);
    cl.def_property_readonly("pmap", &Genten::Sptensor::getProcessorMap, py::return_value_policy::reference, R"(
    Processor map for distributed memory parallelism.)");
    cl.def_property_readonly("ndims", &Genten::Sptensor::ndims, R"(
    Number of dimensions of the sptensor.)");
    cl.def_property_readonly("shape", [](const Genten::Sptensor& X) {
        const ttb_indx nd = X.ndims();
        auto sz = py::tuple(nd);
        for (ttb_indx i=0; i<nd; ++i)
          sz[i] = X.size(i);
        return sz;
      }, R"(
    The dimensions of the sptensor.)");
    cl.def_property_readonly("subs",[](const Genten::Sptensor& X) {
        auto subs = X.getSubscripts();
        using subs_type = decltype(subs);
        py::capsule capsule(new subs_type(subs), [](void *s) { delete reinterpret_cast<subs_type*>(s); });
        const ttb_indx nd = X.ndims();
        const ttb_indx nz = X.nnz();
        return py::array_t<ttb_indx,py::array::c_style>({nz, nd}, subs.data(), capsule);
      }, R"(
    The 2-D numpy.ndarray of nonzero coordinates.)");
    cl.def_property_readonly("vals",[](const Genten::Sptensor& X) {
        Genten::Array vals = X.getValues();
        py::capsule capsule(new Genten::Array(vals), [](void *v) { delete reinterpret_cast<Genten::Array*>(v); });
        const pybind11::ssize_t nz = X.nnz();
        return py::array_t<ttb_real,py::array::c_style>(nz, vals.ptr(), capsule);
      }, R"(
    The 1-D numpy.ndarray of nonzero values.)");
    cl.def_property_readonly("nnz", [](const Genten::Sptensor& X) {
        return X.nnz();
      }, R"(
    Returns the total number of elements in the sptensor.)");
    cl.def_property_readonly("is_values_view", [](const Genten::Sptensor& X) {
        // X has a values view if it has a pair of buffers for extra data,
        // or one buffer that with ttb_real scalar type (can't use dimension
        // since pyttb passes in values arrays that are 2-D sometimes).
        if (X.has_extra_data()) {
          if (X.has_extra_data_type< std::pair<py::buffer,py::buffer> >())
            return true;
          py::buffer b = X.get_extra_data<py::buffer>();
          py::buffer_info info = b.request();
          if (info.format == py::format_descriptor<ttb_real>::format())
            return true;
        }
        return false;
      }, R"(
    Return whether whether the values array is a view of a numpy array (i.e.,
    constructed with copy=False).)");
    cl.def_property_readonly("is_subs_view", [](const Genten::Sptensor& X) {
        // X has a subs view if it has a pair of buffers for extra data,
        // or one buffer that with ttb_indx scalar type (can't use dimension
        // since pyttb passes in values arrays that are 2-D sometimes).
        if (X.has_extra_data()) {
          if (X.has_extra_data_type< std::pair<py::buffer,py::buffer> >())
            return true;
          py::buffer b = X.get_extra_data<py::buffer>();
          py::buffer_info info = b.request();
          if (info.format == py::format_descriptor<ttb_indx>::format() ||
              (info.format == "L" && sizeof(unsigned long) == sizeof(ttb_indx)))
            return true;
        }
        return false;
      }, R"(
    Return whether whether the values array is a view of a numpy array (i.e.,
    constructed with copy=False).)");
    cl.def("__str__", [](const Genten::Sptensor& X) {
        std::stringstream ss;
        Genten::print_sptensor(X, ss);
        return ss.str();
      }, R"(
    Returns a string representation of the sptensor.)");
  }
  {
    py::class_<Genten::DTC, std::shared_ptr<Genten::DTC>> cl(m, "DistTensorContext", R"(
    Class for creating and managing distributed sparse/dense tensors.
    Parallel processor information is not setup until a tensor is distributed
    in parallel by calling one of the 'distributeTensor' methods.)");
    cl.def( py::init( [](){ return new Genten::DTC(); } ), R"(
    Create an empty DistTensorContext.)" );
    cl.def("getProcessorMap", [](const Genten::DTC& dtc) {
        return dtc.pmap_ptr();
      }, R"(
    Return the the parallel processor map.)");
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
        return dtc.distributeTensor(file, index_base, compressed, tree,
                                    algParams);
      }, R"(
    Read a tensor from a file and distribute it in parallel.

    The type of tensor is determined by the file and a tuple of two tensor
    objects is returned, the first a sparse tensor and the second a dense.
    Which type of tensor was read can be determined by determing which of the
    two returned is non-empty.

    Parameters:
      * file:  The name of the file to read.
      * index_base: The starting index for coordinates in the file for sparse
        tensors.  Tensors written by the Matlab Tensor Toolbox have a starting
        index of 1.
      * compressed: Whether the file is compressed or not.
      * json:  A JSON string with optional parameter information.
      * algParams:  Optional parmeter information stored as AlgParams.)", py::arg("file"), py::arg("index_base"), py::arg("compressed"), py::arg("json"), py::arg("algParams"));
    cl.def("distributeTensor",[](Genten::DTC& dtc, const std::string& json, const Genten::AlgParams& algParams) {
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
        return dtc.distributeTensor(tree, algParams);
      }, R"(
    Read a tensor from a JSON file and distribute it in parallel.

    The type of tensor is determined by the file and a tuple of two tensor
    objects is returned, the first a sparse tensor and the second a dense.
    Which type of tensor was read can be determined by determing which of the
    two returned is non-empty.

    Parameters:
      * json:  A JSON string containing information about the tensor.
      * algParams:  Optional parmeter information stored as AlgParams.)", py::arg("json"), py::arg("algParams"));
    cl.def("distributeTensor", [](Genten::DTC& dtc, const Genten::Sptensor& X, const Genten::AlgParams& algParams) {
        return dtc.distributeTensor(X, algParams);
      }, R"(
    Distribute a given sparse tensor in parallel and return the distributed
    tensor.)");
    cl.def("distributeTensor", [](Genten::DTC& dtc, const Genten::Tensor& X, const Genten::AlgParams& algParams) {
        return dtc.distributeTensor(X, algParams);
      }, R"(
    Distribute a given dense tensor in parallel and return the distributed
    tensor.)");
    cl.def("importToAll", [](const Genten::DTC& dtc, const Genten::Ktensor& u) {
        return dtc.importToAll<Genten::DefaultHostExecutionSpace>(u);
      }, R"(
    Import a given Ktensor to all processors.)");
    cl.def("importToRoot", [](const Genten::DTC& dtc, const Genten::Ktensor& u) {
        return dtc.importToRoot<Genten::DefaultHostExecutionSpace>(u);
      }, R"(
    Import a given Ktensor to the root processor.)");
    cl.def("exportFromRoot", [](const Genten::DTC& dtc, const Genten::Ktensor& u) {
        return dtc.exportFromRoot<Genten::DefaultHostExecutionSpace>(u);
      }, R"(
    Export a Ktensor from the root processor to all processors.)");
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
