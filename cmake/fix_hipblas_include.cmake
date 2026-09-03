# hipify-perl emits `#include <hipblas.h>`; ROCm installs the header as <hipblas/hipblas.h>.
file(READ "${file}" _c)
string(REPLACE "<hipblas.h>" "<hipblas/hipblas.h>" _c "${_c}")
file(WRITE "${file}" "${_c}")
