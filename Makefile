CC = g++-5 -Wall
LDFLAGS = -lgsl -lm -lcblas
CXXFLAGS=-fopenmp -fPIC -pipe -pthread -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF

GSL_INCLUDE_MAC = /usr/local/include/
GSL_LIB_MAC = /usr/local/lib/

GSL_INCLUDE_linux = /home/escenter11/gym/gsl/include/
GSL_LIB_linux = /home/escenter11/gym/gsl/lib/

OPEN_BLAS_INCLUDE= /home/lqb/opt/OpenBLAS/include
OPEN_BLAS_LIB= /home/lqb/opt/OpenBLAS/lib

LSOURCE = main.cpp utils.cpp socialexpo.cpp data.cpp eval.cpp
LHEADER = utils.h  socialexpo.h data.h eval.c


sexpo: $(LSOURCE) $(HEADER)
	 $(CC) -I$(GSL_INCLUDE_linux) -I$(OPEN_BLAS_INCLUDE) -L$(GSL_LIB_linux) -L$(OPEN_BLAS_LIB) $(LSOURCE) -o $@ $(LDFLAGS) ${CXXFLAGS}


clean:
	-rm -f *.o sexpo
