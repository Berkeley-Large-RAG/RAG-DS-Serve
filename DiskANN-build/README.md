## DiskANN-build quick setup 

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
cd /home/yichuan_wang/DS-serve/DiskANN-build/DiskANN
pip install -e .
```

## DS-serve build

### Test set setup

```
mkdir -p DiskANN/build/data && cd DiskANN/build/data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd ..
./apps/utils/fvecs_to_bin float data/sift/sift_learn.fvecs data/sift/sift_learn.fbin
./apps/utils/fvecs_to_bin float data/sift/sift_query.fvecs data/sift/sift_query.fbin

./apps/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file data/sift/sift_learn.fbin --query_file  data/sift/sift_query.fbin --gt_file data/sift/sift_query_learn_gt100 --K 100
# Using 0.003GB search memory budget for 100K vectors implies 32 byte PQ compression
./apps/build_disk_index --data_type float --dist_fn l2 --data_path data/sift/sift_learn.fbin --index_path_prefix data/sift/disk_index_sift_learn_R32_L50_A1.2 -R 32 -L50 -B 0.003 -M 1
 ./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/disk_index_sift_learn_R32_L50_A1.2 --query_file data/sift/sift_query.fbin  --gt_file data/sift/sift_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path data/sift/res --num_nodes_to_cache 10000
 ```

### bin to int8 support
```

python3 - << 'PY'
import numpy as np, struct
p='data/sift/sift_learn.fbin'
with open(p,'rb') as f:
    n,d=struct.unpack('<II', f.read(8))
m=np.memmap(p, dtype=np.float32, mode='r', offset=8, shape=(n,d))
mn=float(m.min()); mx=float(m.max())
print(f'N={n} D={d} MIN={mn} MAX={mx} BIAS={(mn+mx)/2} SCALE={(mx-mn)}')
PY

./apps/utils/float_bin_to_int8 data/sift/sift_learn.fbin data/sift/sift_learn.i8bin <BIAS> <SCALE>

```

Actually what I use
```
./apps/utils/float_bin_to_int8 data/sift/sift_learn.fbin data/sift/sift_learn.i8bin 98.5 197.0

./apps/utils/float_bin_to_int8 data/sift/sift_query.fbin data/sift/sift_query.i8bin 98.5 197.0

./apps/build_disk_index \
  --data_type int8 --dist_fn l2 \
  --data_path data/sift/sift_learn.i8bin \
  --index_path_prefix data/sift/disk_index_sift_learn_i8_R32_L50_A1.2 \
  -R 32 -L50 -B 0.003 -M 1


  ./apps/search_disk_index \
  --data_type int8 --dist_fn l2 \
  --index_path_prefix data/sift/disk_index_sift_learn_i8_R32_L50_A1.2 \
  --query_file data/sift/sift_query.i8bin \
  --gt_file data/sift/sift_query_learn_gt100 \
  -K 10 -L 10 20 30 40  \
  --result_path data/sift/res_i8 \
  --num_nodes_to_cache 10000
```


### Actual CompactDS

```
./apps/build_disk_index --data_type float --dist_fn l2 --data_path /mnt/hyperdisk/embeddings_diskann/vectors.bin --index_path_prefix /mnt/hyperdisk/embeddings_diskann/disk_index_compactDS_learn_R60_L80_B180_M800 -R 60 -L80 -B 180 -M 800

```