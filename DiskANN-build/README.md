## DiskANN-build quick setup (Debian 12)

Install base build tools and dependencies:
```bash
sudo apt install -y make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev
```

Conda install MKL and openMP userspace:
```
conda create -n DS-serve python=3.11
conda install -y -c conda-forge -c defaults intel-openmp mkl mkl-include
export OMP_PATH="$CONDA_PREFIX/lib"
export MKL_PATH="$CONDA_PREFIX/lib"
export MKL_INCLUDE_PATH="$CONDA_PREFIX/include"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
cd /home/yichuan_wang/DS-serve/DiskANN-build/DiskANN
rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DOMP_PATH="$OMP_PATH" -DMKL_PATH="$MKL_PATH" -DMKL_INCLUDE_PATH="$MKL_INCLUDE_PATH" ..
make -j
```


For pip install:
```
conda install -y -c conda-forge -c defaults intel-openmp mkl mkl-include cmake ninja scikit-build
export CMAKE_ARGS="-DOMP_PATH=$CONDA_PREFIX/lib -DMKL_PATH=$CONDA_PREFIX/lib -DMKL_INCLUDE_PATH=$CONDA_PREFIX/include -DCMAKE_PREFIX_PATH=$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```
