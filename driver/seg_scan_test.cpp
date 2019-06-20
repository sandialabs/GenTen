#include <string>
#include <iomanip>
#include <iostream>

#include "Genten_Kokkos.hpp"

#include "Kokkos_Random.hpp"

template <typename ValViewType, typename FlagViewType>
void seg_scan(const ValViewType& vals, const FlagViewType& flags,
              const bool check = false)
{
  typedef typename ValViewType::non_const_value_type val_type;
  typedef typename FlagViewType::non_const_value_type flag_type;
  typedef typename ValViewType::size_type size_type;
  typedef typename ValViewType::execution_space exec_space;
  typedef typename Kokkos::TeamPolicy<exec_space>::member_type TeamMember;
  typedef Kokkos::View<val_type*, Kokkos::LayoutRight, typename exec_space::scratch_memory_space , Kokkos::MemoryUnmanaged > TmpScratchSpace;
  typedef Genten::SpaceProperties<exec_space> Prop;

  const size_type n = vals.extent(0);
  const size_type r = vals.extent(1);
  size_type block_size, num_blocks, block_threshold, league_size, team_size,
    vector_size;
  if (Prop::is_cuda) {
    vector_size = r;
    team_size = 256 / vector_size;
    block_size = 32;
    if (n < block_size) {
      block_size = n;
      team_size = 1;
    }
    num_blocks = (n+block_size-1)/block_size;
    league_size = (n+team_size*block_size-1)/(team_size*block_size);
    block_threshold = 1;
  }
  else {
    const size_type num_threads = Prop::concurrency();
    vector_size = r;
    team_size = 1;
    if (n > num_threads) {
      block_size = (n+num_threads-1)/num_threads;
      num_blocks = (n+block_size-1)/block_size;
    }
    else {
      num_blocks = 1;
      block_size = n;
    }
    league_size = num_blocks;
    block_threshold = 8;
  }
  const size_t bytes = TmpScratchSpace::shmem_size(r);

  ValViewType vals_orig;
  if (check) {
    vals_orig = ValViewType("vals_orig",n,r);
    Kokkos::deep_copy(vals_orig, vals);
  }

  if (check)
    std::cout << "n = " << n << " block_size = " << block_size << " num_blocks = " << num_blocks << " league_size = " << league_size << " team_size = " << team_size << std::endl;

  if (num_blocks > block_threshold) {
    // Parallel scan
    ValViewType block_vals("block_vals", num_blocks, r);
    FlagViewType block_flags("block_flags", num_blocks);
    Kokkos::TeamPolicy<exec_space> policy(league_size,team_size,vector_size);
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        if (block >= num_blocks) return;

        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        flag_type f = block == 0 ? 1 : 0;
        for (size_type k=0; k<block_size; ++k) {
          size_type i = block*block_size + k;
          if (i >= n) continue;
          if (flags(i) == 1) {
            f = flag_type(1);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(i,j);
            });
          }
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(i,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) = s[j];
          });
        }
        block_flags(block) = f;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          block_vals(block,j) = s[j];
        });
      });

    // Scan the block results that are in the same segment
    seg_scan(block_vals, block_flags, check);

    // Update scans for blocks [1,num_blocks) from inter-block scans
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block =
          team.league_rank()*team.team_size() + team.team_rank();
        size_type i = block*block_size;
        if (block >= num_blocks || i >= n) return;
        TmpScratchSpace s(team.thread_scratch(0), r);
        if (block == 0) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = block_vals(block-1,j);
        });
        while (i<n && i<(block+1)*block_size && flags(i)==0) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) += s[j];
          });
          ++i;
        }
      });
  }
  else {
    // Serial scan
    Kokkos::TeamPolicy<exec_space> policy(1,1,vector_size);
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        TmpScratchSpace s(team.thread_scratch(0), r);
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = val_type(0);
        });
        for (size_type i=0; i<n; ++i) {
          if (flags(i) == 1)
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] = vals(i,j);
            });
          else
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                                 [&] (const unsigned& j)
            {
              s[j] += vals(i,j);
            });
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                               [&] (const unsigned& j)
          {
            vals(i,j) = s[j];
          });
        }
      });
  }

  if (check) {
    // Check scan is correct
    typename ValViewType::HostMirror vals_host =
      Kokkos::create_mirror_view(vals_orig);
    typename FlagViewType::HostMirror flags_host =
      Kokkos::create_mirror_view(flags);
    typename ValViewType::HostMirror scans_host =
      Kokkos::create_mirror_view(vals);
    Kokkos::deep_copy(vals_host, vals_orig);
    Kokkos::deep_copy(flags_host, flags);
    Kokkos::deep_copy(scans_host, vals);
    bool correct = true;
    std::vector<val_type> s(r);
    for (int i=0; i<n; ++i) {
      if (i == 0 || flags_host(i) == 1)
        for (int j=0; j<r; ++j)
          s[j] = vals_host(i,j);
      else
        for (int j=0; j<r; ++j)
          s[j] += vals_host(i,j);
      for (int j=0; j<r; ++j) {
        if (scans_host(i,j) != s[j]) {
          correct = false;
          break;
        }
      }
    }

    // Print incorrect values
    if (!correct) {
      const int w1 = std::ceil(std::log10(n))+2;
      const val_type w2 = std::ceil(std::log10(100))+2;
      std::cout << std::setw(w1) << "i" << " "
                << std::setw(2) << "f" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2-1) << "v" << j << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << "s" << j << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << "t" << j << " ";
      std::cout << std::endl
                << std::setw(w1) << "==" << " "
                << std::setw(2) << "=" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << "==" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << "==" << " ";
      std::cout << std::endl;
      for (int i=0; i<n; ++i) {
        if (i == 0 || flags_host(i) == 1)
          for (int j=0; j<r; ++j)
            s[j] = vals_host(i,j);
        else
          for (int j=0; j<r; ++j)
            s[j] += vals_host(i,j);
        bool line_correct = true;
        for (int j=0; j<r; ++j) {
          if (scans_host(i,j) != s[j]) {
            line_correct = false;
            break;
          }
        }
        if (!line_correct) {
          std::cout << std::setw(w1) << i << " "
                    << std::setw(2) << flags_host(i) << " ";
          for (int j=0; j<r; ++j)
            std::cout << std::setw(w2) << vals_host(i,j) << " ";
          for (int j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << scans_host(i,j) << " ";
          for (int j=0; j<r; ++j)
            std::cout << std::setw(w2+1) << s[j] << " ";
          std::cout << "Wrong!" << std::endl;
        }
      }
      std::cout << "Scan is not correct!" << std::endl;
    }
    else
      std::cout << "Scan is correct!" << std::endl;
  }
}

typedef Kokkos::DefaultExecutionSpace ExecSpace;
//typedef Kokkos::Serial ExecSpace;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc,argv);
  {

    // Length of vector to scan
    int n = 100;
    if (argc >= 3 && std::string(argv[1]) == "-n")
      n = std::atoi(argv[2]);
    int r = 16;

    typedef int flag_type;
    typedef ptrdiff_t val_type;
    typedef Kokkos::View<flag_type*,ExecSpace> FlagViewType;
    typedef Kokkos::View<val_type**,ExecSpace> ValViewType;
    FlagViewType flags("vals",n,r);
    ValViewType vals("vals",n,r);
    ValViewType scans("scans",n,r);

    // Initialize vals to random values in (-range,range) and flags to 0/1's
    // with an average segment length of seg_length
    typedef Kokkos::Random_XorShift64_Pool<ExecSpace> RandomPool;
    typedef RandomPool::generator_type generator_type;
    typedef Kokkos::rand<generator_type, val_type> RandVal;
    typedef Kokkos::rand<generator_type, val_type> RandFlag;
    const val_type range = 100;
    const flag_type seg_length = 10;
    const int seed = 12345;
    RandomPool rand_pool(seed);
    Kokkos::RangePolicy<ExecSpace> policy(0,n);
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i)
    {
      generator_type gen = rand_pool.get_state();
      for (int j=0; j<r; ++j)
        vals(i,j) = RandVal::draw(gen,-range,range);
      flags(i) =
        ((i > 0) && (RandFlag::draw(gen,0,seg_length+1) < seg_length)) ? 0 : 1;
      rand_pool.free_state(gen);
    });

    // Now do segmented scan
    Kokkos::deep_copy(scans, vals);
    seg_scan(scans, flags, true);

    // Print and check results
#if 0
    ValViewType::HostMirror vals_host = Kokkos::create_mirror_view(vals);
    FlagViewType::HostMirror flags_host = Kokkos::create_mirror_view(flags);
    ValViewType::HostMirror scans_host = Kokkos::create_mirror_view(scans);
    Kokkos::deep_copy(vals_host, vals);
    Kokkos::deep_copy(flags_host, flags);
    Kokkos::deep_copy(scans_host, scans);
    const int w1 = std::ceil(std::log10(n))+2;
    const val_type w2 = std::ceil(std::log10(range))+2;
    std::cout << std::setw(w1) << "i" << " "
              << std::setw(2) << "f" << " ";
    for (int j=0; j<r; ++j)
      std::cout << std::setw(w2-1) << "v" << j << " ";
    for (int j=0; j<r; ++j)
      std::cout << std::setw(w2) << "s" << j << " ";
    for (int j=0; j<r; ++j)
      std::cout << std::setw(w2) << "t" << j << " ";
    std::cout << std::endl
              << std::setw(w1) << "==" << " "
              << std::setw(2) << "=" << " ";
     for (int j=0; j<r; ++j)
       std::cout << std::setw(w2) << "==" << " ";
     for (int j=0; j<r; ++j)
       std::cout << std::setw(w2+1) << "==" << " ";
     for (int j=0; j<r; ++j)
       std::cout << std::setw(w2+1) << "==" << " ";
     std::cout << std::endl;
    bool correct = true;
     std::vector<val_type> s(r);
    for (int i=0; i<n; ++i) {
      if (i == 0 || flags_host(i) == 1)
        for (int j=0; j<r; ++j)
          s[j] = vals_host(i,j);
      else
        for (int j=0; j<r; ++j)
          s[j] += vals_host(i,j);
      std::cout << std::setw(w1) << i << " "
                << std::setw(2) << flags_host(i) << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2) << vals_host(i,j) << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << scans_host(i,j) << " ";
      for (int j=0; j<r; ++j)
        std::cout << std::setw(w2+1) << s[j] << " ";
      for (int j=0; j<r; ++j) {
        if (scans_host(i,j) != s[j]) {
          std::cout << "Wrong!";
          correct = false;
          break;
        }
      }
      std::cout << std::endl;
    }
    if (correct)
      std::cout << "Passed!" << std::endl;
    else
      std::cout << "Failed!" << std::endl;
#endif

  }
  Kokkos::finalize();

  return 0;
}
