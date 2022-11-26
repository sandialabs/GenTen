Builds a container for GenTen on OLCF/Summit.

Run build.sh from one level above the top-level genten directory to build
the container.

The singularity build is run on a compute node, so the top-level genten
directory needs to be in the $MEMBERWORK area to have write access

Once the container is built, run an example GenTen decomposition with
```
bsub genten/docker/olcf_summit/run-genten-example.lsf
```
from the same location.
