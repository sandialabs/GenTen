#include <string>
#include <iomanip>
#include <iostream>

#include "Genten_Kokkos.hpp"

#include "Kokkos_Random.hpp"

// Serial segmented scan within a block
template <typename ValViewType, typename FlagViewType,
          typename TeamMember, typename size_type>
KOKKOS_INLINE_FUNCTION
typename FlagViewType::non_const_value_type
seg_scan_block(const ValViewType& vals,
               const FlagViewType& flags,
               const TeamMember& team,
               const size_type n,
               const size_type r,
               typename ValViewType::non_const_value_type* res)
{
  // Flag result of the intra-block scan, which is the OR of the flags across
  // the block
  typedef typename FlagViewType::non_const_value_type flag_type;
  flag_type f = flag_type(0);
  auto vector_policy = Kokkos::ThreadVectorRange(team, r);
  for (size_type i=1; i<n; ++i) {
    if (flags(i) != 1)
      Kokkos::parallel_for(vector_policy, [&](const unsigned j)
      {
        vals(i,j) += vals(i-1,j);
      });
    else
      f = flag_type(1);
  }
  // Result of the intra-block scan
  if (res != nullptr)
    Kokkos::parallel_for(vector_policy, [&](const unsigned j)
    {
      res[j] = vals(n-1,j);
    });
  return f;
}

// Segmented scan within a team
template <typename ValViewType, typename FlagViewType,
          typename TeamMember, typename size_type>
void
seg_scan_team(const ValViewType& vals,
              const FlagViewType& flags,
              const ValViewType& block_vals,
              const FlagViewType& block_flags,
              const TeamMember& team,
              const size_type thread_block_size)
{
  typedef typename ValViewType::non_const_value_type val_type;
  typedef typename FlagViewType::non_const_value_type flag_type;
  typedef typename ValViewType::execution_space exec_space;
  typedef Kokkos::View<val_type**, Kokkos::LayoutRight, typename exec_space::scratch_memory_space , Kokkos::MemoryUnmanaged > ValScratchSpace;
  typedef Kokkos::View<flag_type*, Kokkos::LayoutRight, typename exec_space::scratch_memory_space , Kokkos::MemoryUnmanaged > FlagScratchSpace;

  const size_type team_rank = team.team_rank();
  const size_type team_size = team.team_size();
  const size_type team_block_size = thread_block_size * team_size;
  const size_type n = vals.extent(0);
  const size_type r = vals.extent(1);
  const size_type num_blocks = block_vals.extent(0);
  const size_type block =
    team.league_rank()*team.team_size() + team.team_rank();
  ValScratchSpace val_scratch(team.team_scratch(0), team_block_size, r);
  ValScratchSpace team_vals(team.team_scratch(0), team_size, r);
  FlagScratchSpace flag_scratch(team.team_scratch(0), team_block_size);
  FlagScratchSpace team_flags(team.team_scratch(0), team_size);

  // Load vals and flags into scratch
  for (size_type k=0; k<thread_block_size; ++k) {
    const size_type i_global = block*thread_block_size + k;
    const size_type i_local = team_rank*thread_block_size + k;
    if (i_global >= n) break;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                         [&] (const unsigned j)
    {
      val_scratch(i_local,r) = vals(i_global,r);
    });
    flag_scratch(i_local) = flags(i_global);
  }

  // Do block scan
  const size_type n_block = (block+1)*thread_block_size <= n ?
    thread_block_size : n - block*thread_block_size;
  team_flags(team_rank) =
    seg_scan_block(val_scratch, flag_scratch, team, n_block, r,
                   &team_vals(team_rank,0));
  team.team_barrier();

  // Do scan of team results
  if (team_rank == 0)
    block_flags(team.league_rank()) =
      seg_scan_block(team_vals, team_flags, team, team_size, r,
                     &block_vals(team.league_rank(),0));
  team.team_barrier();

  // Incorporate team scans
  if (team_rank > 0) {
    for (size_type k=0; k<thread_block_size; ++k) {
      const size_type i_global = block*thread_block_size + k;
      const size_type i_local = team_rank*thread_block_size + k;
      if (i_global >= n) break;
      if (flag_scratch(i_local) == 1) break;
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                           [&] (const unsigned j)
      {
        val_scratch(i_local,r) =  team_vals(team_rank-1,j);
      });
    }
  }
}

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
  size_type thread_block_size, block_threshold, league_size, team_size,
    vector_size;
  if (Prop::is_cuda) {
    vector_size = r;
    team_size = 256 / vector_size;
    thread_block_size = 32;
    if (n < thread_block_size) {
      thread_block_size = n;
      team_size = 1;
    }
    league_size =
      (n+team_size*thread_block_size-1)/(team_size*thread_block_size);
    block_threshold = 1;
  }
  else {
    const size_type num_threads = Prop::concurrency();
    vector_size = r;
    team_size = 1;
    if (n > num_threads) {
      thread_block_size = (n+num_threads-1)/num_threads;
      league_size = (n+thread_block_size-1)/thread_block_size;
    }
    else {
      league_size = 1;
      thread_block_size = n;
    }
    block_threshold = 8;
  }
  const size_t bytes = TmpScratchSpace::shmem_size(r);

  ValViewType vals_orig;
  if (check) {
    vals_orig = ValViewType("vals_orig",n,r);
    Kokkos::deep_copy(vals_orig, vals);
  }

  if (check)
    std::cout << "n = " << n << " thread_block_size = " << thread_block_size << " league_size = " << league_size << " team_size = " << team_size << std::endl;

  if (league_size > block_threshold) {
    // Parallel scan
    ValViewType block_vals("block_vals", league_size, r);
    FlagViewType block_flags("block_flags", league_size);
    Kokkos::TeamPolicy<exec_space> policy(league_size,team_size,vector_size);
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        seg_scan_team(vals, flags, block_vals, block_flags, team,
                      thread_block_size);
      });

    // Scan the block results that are in the same segment
    seg_scan(block_vals, block_flags, check);

    // Update scans for blocks [1,num_blocks) from inter-block scans
    Kokkos::parallel_for(
      policy.set_scratch_size(0,Kokkos::PerThread(bytes)),
      KOKKOS_LAMBDA(const TeamMember& team)
      {
        const size_type block = team.league_rank();
        const size_type team_size = team.team_size();
        size_type i = block*thread_block_size*team_size;
        if (i >= n) return;
        TmpScratchSpace s(team.thread_scratch(0), r);
        if (block == 0) return;
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, r),
                             [&] (const unsigned& j)
        {
          s[j] = block_vals(block-1,j);
        });
        while (i<n && i<(block+1)*thread_block_size*team_size && flags(i)==0) {
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
    int r = 1;

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
