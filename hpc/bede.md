# Fes-ml Installation on Bede (POWER9 System, ppc64le)

## Create the Base Conda Environment

This will install Python, PyTorch, cudatoolkit, OpenFF, ASE, TorchANI, EMLE, among other packages.

```bash
conda config --set channel_priority flexible
conda env create -f environment_ppc64le.yaml
conda config --set channel_priority strict
conda activate fes-ml
```

## Compile Software from the OpenMM Ecosystem

1. Load Modules and Install Necessary Packages to Compile OpenMM:

```bash
module load cuda/12.0.1
conda install -c conda-forge swig doxygen cython
```

2. Compile OpenMM:

```bash
git clone https://github.com/openmm/openmm.git
cd openmm
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make -j32 install
make PythonInstall
python -m openmm.testInstallation
```

3. Compile NNPOps:

```bash
git clone https://github.com/openmm/NNPOps.git
cd NNPOps
mkdir build && cd build
cmake .. \
    -DTorch_DIR=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')/Torch \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j32 install
```

**Note:** If you encounter errors like `error: expected template-name before ‘<’ token`, downgrade gcc to version 11.4.0:

```bash
conda install -c conda-forge gcc=11.4.0
```

Then, delete the contents of the `build` directory and recompile.

4. Compile openmm-torch:

```bash
git clone https://github.com/openmm/openmm-torch
cd openmm-torch
mkdir build && cd build
ccmake ..
```

Executing `ccmake ..` will open a configuration menu. Press `c` to configure and change the fields to the following:

- `CMAKE_INSTALL_PREFIX`: Output of `$CONDA_PREFIX`
- `OPENMM_DIR`: Output of `$CONDA_PREFIX`
- `PYTORCH_DIR`: `$CONDA_PREFIX/lib/python3.10/site-packages/torch`
- `Torch_DIR`: Output of `python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)'`

Press `g` to generate the installation configuration. Then compile openmm-torch:

```bash
make -j32 install
make PythonInstall
```

For OpenMM to load the OpenMMTorch kernels successfully, add the path where LibTorch is installed to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

5. Compile openmm-ml:

```bash
git clone https://github.com/openmm/openmm-ml.git
cd openmm-ml
pip install .
```

6. Compile MACE if you want to use that MLP with openmm-ml:

```bash
pip install mace-torch
```

At this stage, you have a fully-fledged OpenMM environment installation that can run hybrid ML/MM simulations.



## Compile RDKit and AmberTools (Dependencies of Sire and OpenFF required for fes-ml to work)

Before attempting to install anything else, install the Boost libraries by executing:

```bash
conda install -c conda-forge boost boost-cpp
```

1. Compile RDKit:

```bash
git clone https://github.com/rdkit/rdkit.git
cd rdkit
mkdir build && cd build
cmake -DPy_ENABLE_SHARED=1 \
  -DRDK_INSTALL_INTREE=OFF \
  -DRDK_INSTALL_STATIC_LIBS=OFF \
  -DRDK_BUILD_CPP_TESTS=ON \
  -DPYTHON_NUMPY_INCLUDE_PATH="$(python -c 'import numpy ; print(numpy.get_include())')" \
  -DBOOST_DIR=$CONDA_PREFIX \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  ..
make -j32 install
```

2. Compile AmberTools:

First, install the X11 libraries:

```bash
conda install -c conda-forge xorg-libxt xorg-libx11 xorg-libxrender xorg-libxext
```

Then, download AmberTools from [AmberMD](https://ambermd.org/GetAmber.php). Once you've got `AmberTools24.tar.bz2` on Bede, execute the following commands:

```bash
tar xjfv AmberTools24.tar.bz2
cd amber24_src/build
vim run_cmake
```

Make sure to set:

```bash
-DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX/amber24
-DDOWNLOAD_MINICONDA=FALSE
```

Then execute:

```bash
./run_cmake
make -j32 install
```

Finally, change to the directory where AmberTools was installed and source the Amber environment:

```bash
cd $CONDA_PREFIX/amber24
source amber.sh
```

Consider sourcing this file in your `.bashrc` to automatically have the Amber environment ready for production:

```bash
echo "source $CONDA_PREFIX/amber24/amber.sh" >> ~/.bashrc
```

## Compile Sire

First, install as many additional dependencies as possible:

```bash
conda install -c conda-forge tbb tbb-devel libnetcdf gsl rich zlib
pip install gemmi lazy_import
```

Most of the other dependencies were already included in the base conda environment. Then proceed to compile Sire:

```bash
git clone https://github.com/openbiosim/sire
cd sire
git checkout feature_emle
export Torch_DIR=$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')/Torch
python setup.py install --skip-deps -N 32
```

## Install fes-ml

```bash
git clone git@github.com:michellab/fes-ml.git
cd fes-ml
pip install -e .
```

Fix some dependencies:

```bash
pip install networkx==3.3
```

If necessary, reinstall emle-engine using the `feature_aev` branch:

```bash
git clone https://github.com/chemle/emle-engine.git
cd emle-engine
git checkout feature_aev
pip install -e .
```
