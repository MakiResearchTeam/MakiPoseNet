%module pafprocess
%{
  #define SWIG_FILE_WITH_INIT
  #include "pafprocess.h"
%}

%include "numpy.i"
%init %{
import_array();
%}


%apply (int DIM1, float* IN_ARRAY1) {(int size_score_from_peaks, float *score_from_peaks)}
%apply (int DIM1, int DIM2, int* IN_ARRAY2) {(int p1, int p2, int *peak_info_data)}
%apply (int DIM1, int DIM2, int DIM3, float* IN_ARRAY3) {(int f1, int f2, int f3, float *pafmap)};
%include "pafprocess.h"
