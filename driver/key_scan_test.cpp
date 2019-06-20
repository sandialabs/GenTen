#include <string>
#include <iomanip>
#include <iostream>

#include "Genten_KokkosAlgs.hpp"

#include "Kokkos_Random.hpp"
#include "Kokkos_Sort.hpp"

typedef Kokkos::DefaultExecutionSpace ExecSpace;
//typedef Kokkos::Serial ExecSpace;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc,argv);
  {

    // Length of vector to scan
    int n = 100;
    if (argc >= 3 && std::string(argv[1]) == "-n")
      n = std::atoi(argv[2]);
    int r = 2;

    typedef int key_type;
    typedef int val_type;
    typedef int perm_type;
    typedef Kokkos::View<key_type*,ExecSpace> KeyViewType;
    typedef Kokkos::View<val_type**,ExecSpace> ValViewType;
    typedef Kokkos::View<perm_type*,ExecSpace> PermViewType;
    KeyViewType keys("keys",n);
    ValViewType vals("vals",n,r);
    ValViewType scans("scans",n,r);
    KeyViewType perm("perm",n);

    // Initialize vals to random values in (-val_range,val_range) and keys
    // to (0,key_range) so that we get about seg_length keys of each type
    typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
    typedef RandomPool::generator_type generator_type;
    typedef Kokkos::rand<generator_type, val_type> RandVal;
    typedef Kokkos::rand<generator_type, key_type> RandKey;
    const val_type val_range = 100;
    const int seg_length = 10;
    const key_type key_range = n / seg_length;
    const int seed = 12345;
    RandomPool rand_pool(seed);
    Kokkos::RangePolicy<ExecSpace> policy(0,n);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i)
    {
      generator_type gen = rand_pool.get_state();
      for (int j=0; j<r; ++j)
        vals(i,j) = RandVal::draw(gen,-val_range,val_range);
      keys(i) = RandKey::draw(gen,0,key_range);
      rand_pool.free_state(gen);
    });

    // Compute permutation array for sorted keys
    Genten::perm_sort(perm, keys);

    // Now do segmented scan by key
    Kokkos::deep_copy(scans, vals);
    Genten::key_scan(scans, keys, perm, true);

    // Print keys, values, and scans
    const bool print = false;
    if (print) {
      typename ValViewType::HostMirror vals_host =
        Kokkos::create_mirror_view(vals);
      typename KeyViewType::HostMirror keys_host =
        Kokkos::create_mirror_view(keys);
      typename ValViewType::HostMirror scans_host =
        Kokkos::create_mirror_view(scans);
      typename PermViewType::HostMirror perm_host =
        Kokkos::create_mirror_view(perm);
      Kokkos::deep_copy(vals_host, vals);
      Kokkos::deep_copy(keys_host, keys);
      Kokkos::deep_copy(scans_host, scans);
      Kokkos::deep_copy(perm_host, perm);
      std::vector<val_type> s(r);
      const int w1 = std::ceil(std::log10(n))+2;
      const val_type w2 = std::ceil(std::log10(val_range))+2;
      std::cout << std::setw(w1) << "i" << " "
                << std::setw(w2-1) << " k" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2-1) << "v" << j << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << "s" << j << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << "t" << j << " ";
      std::cout << std::endl
                << std::setw(w1) << "==" << " "
                << std::setw(w2) << "==" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << "==" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      std::cout << std::endl;
      key_type key = 0;
      key_type key_prev = 0;
      perm_type p = 0;
      for (int i=0; i<n; ++i) {
        p = perm(i);
        key_prev = key;
        key = keys(p);
        if (p == 0 || key != key_prev)
          for (int j=0; j<r; ++j)
            s[j] = vals_host(p,j);
        else
          for (int j=0; j<r; ++j)
            s[j] += vals_host(p,j);
        bool line_correct = true;
        for (int j=0; j<r; ++j) {
          if (scans_host(i,j) != s[j]) {
            line_correct = false;
            break;
          }
        }
        std::cout << std::setw(w1) << p << " "
                  << std::setw(w2) << keys_host(p) << " ";
        for (int j=0; j<r; ++j)
          std::cout << std::setw(w2) << vals_host(p,j) << " ";
        for (int j=0; j<r; ++j)
          std::cout << std::setw(w2+1) << scans_host(i,j) << " ";
        for (int j=0; j<r; ++j)
          std::cout << std::setw(w2+1) << s[j] << " ";
        if (!line_correct) {
          std::cout << "Wrong!";
        }
        std::cout << std::endl;
      }
    }

    // Reorder keys and values based on perm to use segmented scan
    KeyViewType sorted_keys("sorted_keys",n);
    ValViewType sorted_vals("sorted_vals",n,r);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i)
    {
      const perm_type p = perm(i);
      sorted_keys(i) = keys(p);
      for (int j=0; j<r; ++j)
        sorted_vals(i,j) = vals(p,j);
    });

    // Convert keys to flags and do segmented scan
    typedef int flag_type;
    typedef Kokkos::View<key_type*,ExecSpace> FlagViewType;
    FlagViewType flags("flags",n);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i)
    {
      if (i == 0 || sorted_keys(i) != sorted_keys(i-1))
        flags(i) = flag_type(1);
      else
        flags(i) = flag_type(0);
    });
    ValViewType scans2("scans2",n,r);
    Kokkos::deep_copy(scans2, sorted_vals);
    Genten::seg_scan(scans2, flags, false);

    // Check scans agree
    int num_diff = 0;
    Kokkos::parallel_reduce(policy, KOKKOS_LAMBDA(const int i, int& nd)
    {
      const perm_type p = perm(i);
      for (int j=0; j<r; ++j)
        if (scans(p,j) != scans2(i,j))
          ++nd;
    }, num_diff);
    if (num_diff == 0)
      std::cout << "Key and segmented scans agree!" << std::endl;
    else
      std::cout << "Key and segmented scans do not agree!" << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
