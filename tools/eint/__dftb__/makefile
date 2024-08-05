all:
	gfortran -dynamiclib -O2 -fPIC -o _dftb.so dftb.f90 \
		-Idftbplus-24.1/build/_install/include/dftbplus/modfiles \
		dftbplus-24.1/build/_install/lib/libdftbplus.a \
		dftbplus-24.1/build/_install/lib/libs-dftd3.a \
		dftbplus-24.1/build/_install/lib/libmctc-lib.a \
		/Users/smarti/Applications/miniconda3/envs/chem/lib/libmkl_intel_lp64.dylib \
		/Users/smarti/Applications/miniconda3/envs/chem/lib/libmkl_intel_thread.dylib \
		/Users/smarti/Applications/miniconda3/envs/chem/lib/libmkl_core.dylib \
		/Users/smarti/Applications/miniconda3/envs/chem/lib/libiomp5.dylib -lm -ldl
