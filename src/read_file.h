#ifndef __READ_FILE_H__
#define __READ_FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <random>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
using namespace std;

#include <sli/stdstreamio.h>
#include <sli/tarray_tstring.h>
#include <sli/asarray_tstring.h>
#include <sli/tstring.h>
using namespace sli;

#include <Eigen/Core>
using namespace Eigen;

#include "arg_option.h"

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;

unsigned int
get_data_length(const char*);

unsigned int
get_max(size_t*, size_t*);

unsigned int
get_feature_length(const char*, 
		   tstring, tstring, tstring);

void 
load_data(RMatrixXf*, RowVectorXf*,
	  char*, 
	  tstring, tstring, tstring);

void
show_data_mat(RMatrixXf*);

void
show_data_vec(RowVectorXf*);


#endif //__READ_FILE_H__
