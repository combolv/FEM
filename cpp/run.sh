swig -c++ -python py_fem_lib.i

# Compile c++ code.
mkdir -p build
cd build
cmake ..
make -j4

mv libpy_fem_lib.so ../_fem_lib.so