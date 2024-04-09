#include "Genten_Pybind11_include.hpp"
#include "Genten_DistContext.hpp"
#include "Genten_IOtext.hpp"
#include "Genten_Driver.hpp"
#include "Kokkos_Core.hpp"
#include "Genten_Tpetra.hpp"

#include <pybind11/iostream.h>

namespace py = pybind11;
namespace Genten {
  using DTC = DistTensorContext<Genten::DefaultHostExecutionSpace>;
  using TMap = Genten::tpetra_map_type<Genten::DefaultHostExecutionSpace>;
  using TMV = Genten::tpetra_multivector_type<Genten::DefaultHostExecutionSpace>;
  using Import = Genten::tpetra_import_type<Genten::DefaultHostExecutionSpace>;
}

PYBIND11_MODULE(_phys_utils, m) {
  m.doc() = R"(
     Module providing Python wrappers for physics specific tools.)";
    m.def("global_scalar_min", [](const double& value) -> double {
        gt_assert(Genten::DistContext::initialized());
        double min = 0.0;
        MPI_Allreduce(&value, &min, 1, MPI_DOUBLE, MPI_MIN, Genten::DistContext::commWorld());
	return min;
    }, R"(
    Return minimum scalar over all MPI ranks.)", pybind11::arg("value"));
    m.def("global_go_sum", [](const long long& value) -> long long {
        gt_assert(Genten::DistContext::initialized());
        long long sum = 0;
        MPI_Allreduce(&value, &sum, 1, MPI_LONG_LONG, MPI_SUM, Genten::DistContext::commWorld());
	return sum;
    }, R"(
    Return minimum scalar over all MPI ranks.)", pybind11::arg("value"));
    m.def("get_all_remote_ints", [](const int &value) -> py::array_t<int> {
        gt_assert(Genten::DistContext::initialized());
	int num_procs = Genten::DistContext::nranks();
        int rank      = Genten::DistContext::rank(); 	
	std::vector<int> local_values(num_procs,0);
	local_values[rank] = value;
	std::vector<int> remote_values(num_procs,0);
        MPI_Allreduce(&local_values[0], &remote_values[0], num_procs, MPI_INT, MPI_SUM, Genten::DistContext::commWorld());
	py::array_t<int> output(num_procs, &remote_values[0]);
	return output;
    }, R"(
    Take an integer value from each proc and return an array of these values on all procs)", pybind11::arg("value"));
    m.def("assign_owned_lids", [](py::array_t<Genten::tpetra_go_type> &gids) -> py::array_t<Genten::tpetra_lo_type> {
        gt_assert(Genten::DistContext::initialized());
	Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::MpiComm<int>(Genten::DistContext::commWorld()));
	auto map = Teuchos::rcp(new const Genten::TMap(Teuchos::OrdinalTraits<Genten::tpetra_go_type>::invalid(),gids.data(),gids.size(),0,comm));
	auto one_to_one_map = Tpetra::createOneToOne(map);
	auto total_nodes = one_to_one_map->getGlobalNumElements();
	std::vector<Genten::tpetra_lo_type> owned_lids_vec;;
	for(std::size_t i =0; i < gids.size(); i++)
	{
          if(one_to_one_map->isNodeGlobalElement(gids.at(i)))
            owned_lids_vec.push_back(i);
	}
	py::array_t<Genten::tpetra_lo_type> owned_lids(owned_lids_vec.size(),&owned_lids_vec[0]);
	return owned_lids;
    }, R"(
    Assign each GID to a unique processor and return a list of the LIDs owned by this proc.)", pybind11::arg("gids"));
    m.def("get_reference_ids", [](py::array_t<Genten::tpetra_go_type> &ref_gids) -> py::array_t<Genten::tpetra_go_type> {
        gt_assert(Genten::DistContext::initialized());
	Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::MpiComm<int>(Genten::DistContext::commWorld()));
	auto ref_gid_map = Teuchos::rcp(new Genten::TMap(Teuchos::OrdinalTraits<Genten::tpetra_go_type>::invalid(),ref_gids.data(),ref_gids.size(),0,comm));
	auto rid_map = Teuchos::rcp(new Genten::TMap(ref_gid_map->getGlobalNumElements(),ref_gid_map->getLocalNumElements(),0,comm));
        auto ref_rids_view = rid_map->getMyGlobalIndices();
        py::array_t<Genten::tpetra_go_type> ref_rids(ref_rids_view.extent(0),&ref_rids_view(0));
	return ref_rids;
    }, R"(
    Create new unique IDs from 0 to num_nodes for a set of GIDs.)", pybind11::arg("ref_gids"));
    m.def("broadcast_go_data", [](int &root, py::array_t<Genten::tpetra_go_type> &data) -> void {
        gt_assert(Genten::DistContext::initialized());
        MPI_Bcast(data.mutable_data(), (int) data.size(), MPI_LONG_LONG, root, Genten::DistContext::commWorld());
    }, R"(
    Broadcast an array of global ordinal data from a given root proc to all MPI ranks.)", pybind11::arg("root"), pybind11::arg("data"));
    m.def("broadcast_scalar_data", [](int &root, py::array_t<double> &data) -> void {
        gt_assert(Genten::DistContext::initialized());
        MPI_Bcast(data.mutable_data(), (int) data.size(), MPI_DOUBLE, root, Genten::DistContext::commWorld());
    }, R"(
    Broadcast an array of scalar data from a given root proc to all MPI ranks.)", pybind11::arg("root"), pybind11::arg("data"));
    m.def("redistribute_data_across_procs", [](py::array_t<Genten::tpetra_go_type> &src_ids, py::array_t<double> &src_data, py::array_t<Genten::tpetra_go_type> &dest_ids) -> py::array_t<double> {
        gt_assert(Genten::DistContext::initialized());
	Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::rcp(new Teuchos::MpiComm<int>(Genten::DistContext::commWorld()));
	auto src_map = Teuchos::rcp(new const Genten::TMap(Teuchos::OrdinalTraits<Genten::tpetra_go_type>::invalid(),src_ids.data(),src_ids.size(),0,comm));
	auto dest_map = Teuchos::rcp(new const Genten::TMap(Teuchos::OrdinalTraits<Genten::tpetra_go_type>::invalid(),dest_ids.data(),dest_ids.size(),0,comm));
        auto import = Teuchos::rcp(new const Genten::Import(src_map,dest_map));
	std::size_t num_vecs = src_data.shape(1);
        auto src_mv = Teuchos::rcp(new Genten::TMV(src_map,num_vecs));
	for(std::size_t i=0; i<src_mv->getLocalLength(); i++)
	{
	  for(std::size_t j=0; j<num_vecs; j++)
            src_mv->replaceLocalValue(i,j,src_data.at(i,j));
	}
        auto dest_mv = Teuchos::rcp(new Genten::TMV(dest_map,num_vecs));
        dest_mv->doImport(*src_mv,*import,Tpetra::REPLACE);
        auto dest_view = dest_mv->getLocalViewHost (Tpetra::Access::ReadOnlyStruct());
        py::array_t<double> dest_data({dest_mv->getLocalLength(),num_vecs});
        auto dest_access = dest_data.mutable_unchecked<2>();
	for(std::size_t i=0; i<dest_mv->getLocalLength(); i++)
	{
	  for(std::size_t j=0; j<num_vecs; j++)
            dest_access(i,j) = dest_view(i,j);
	}
	return dest_data;
    }, R"(
    Take an array of data with unique ids on each proc and redistribute the data given a new id distribution)", pybind11::arg("src_ids"), pybind11::arg("src_data"), pybind11::arg("dest_ids"));
    m.def("global_int_array_sum", [](const py::array_t<int> &array) -> py::array_t<int> {
        gt_assert(Genten::DistContext::initialized());
        py::array_t<int> sum(array.size());
        MPI_Allreduce(array.data(), sum.mutable_data(), array.size(), MPI_INT, MPI_SUM, Genten::DistContext::commWorld());
	return sum;
    }, R"(
    Return minimum scalar over all MPI ranks.)", pybind11::arg("value"));
}
