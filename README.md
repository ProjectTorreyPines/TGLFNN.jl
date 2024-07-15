# TGLFNN.jl

Run TGLFNN, TGLF, and QLGYRO models

NOTE: Running TGLF and QLGYRO requires GACODE executables to be locally installed.

Link to instructions on GACODE installation: https://fuse.help/install.html#Install-GACODE

See the note following step 6 - you may need to replace mpif90-openmpi-mp with mpif90-openmpi-gcc12 in the platform-specific make file found in $GACODE_ROOT/platform/build/make.inc.OSX_MONTEREY and mpirun-openmpi-mp with mpirun-openmpi-gcc12 in the platform exec file found in $GACODE_ROOT/platform/exec/exec.OSX_MONTEREY.

## Online documentation
For more details, see the [online documentation](https://projecttorreypines.github.io/TGLFNN.jl/dev).

![Docs](https://github.com/ProjectTorreyPines/TGLFNN.jl/actions/workflows/make_docs.yml/badge.svg)
