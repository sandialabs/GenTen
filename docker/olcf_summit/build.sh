# Needs to be run from one level above the top-level genten directory.
# For building the singularity image on a compute node, needs to
# be in the $MEMBERWORK area
set -x
source genten/docker/olcf_summit/setup_env.sh

podman build -v $MPI_ROOT:$MPI_ROOT -f genten/docker/olcf_summit/Dockerfile -t genten:latest --format=docker genten
podman save -o gentenimage.tar localhost/genten:latest

#singularity build --disable-cache gentenimage.sif docker-archive://gentenimage.tar
# Run singularity build on a compute node
bsub genten/docker/olcf_summit/singularity-build.lsf
