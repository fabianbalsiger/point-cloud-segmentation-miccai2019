#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda  # path to your CUDA directory (you can use $ which nvcc to locate the directory)
TF_PATH=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')  # gets your TensorFlow path (something like /home/fbalsiger/envs/pc/lib/python3.6/site-packages/tensorflow)

${CUDA_PATH}/bin/nvcc -std=c++11 -c -o tf_sampling_g.cu.o tf_sampling_g.cu -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I ${TF_PATH}/include -I ${CUDA_PATH}/include -I ${TF_PATH}/include/external/nsync/public -lcudart -L ${CUDA_PATH}/lib64/ -L ${TF_PATH} -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# see here: https://www.tensorflow.org/guide/extend/op
# As explained above, if you are compiling with gcc>=5 add --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" to the bazel command line.


# comment on the above line: some people mentioned that an error similar to
# tensorflow.python.framework.errors_impl.NotFoundError: (...) tf_sampling_so.so: undefined symbol: _ZTIN10tensorflow8OpKernelE occured
# occured when they run the Python code after compilation. Apparently, removing "-D_GLIBCXX_USE_CXX11_ABI=0" solves the
# issue. But in my case, by removing this flag the error appeared.
# see e.g. https://github.com/charlesq34/pointnet2/issues/48