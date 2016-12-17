#include "arg_option.h"
#include "read_file.h"

unsigned int 
get_data_length(const char *filename)
{
  //read file
  stdstreamio f_in;
  tstring line;

  // file except
  if ( f_in.open("r", filename) < 0 ){
    printf("Can not open %s\n", filename);
    exit(1);
  }
  
  unsigned int count = 0;
  while ( (line = f_in.getline()) != NULL  ) {
    count += 1;
  }

  return count;
}

unsigned 
get_max(size_t* a, size_t* b)
{
  if (a > b) {
    return *a;
  } else {
    return *b;
  }
}

unsigned int 
get_feature_length(const char *filename, 
		   tstring line_delimiter, 
		   tstring line_delimiter_between, 
		   tstring line_delimiter_within)
{
  stdstreamio f_in;
  tstring line;
  tarray_tstring my_arr;
  // file except
  if ( f_in.open("r", filename) < 0 ){
    fprintf(stderr, "Can not open %s\n", filename);
    exit(1);
  }

  size_t max_feature_id=0;
  size_t feature_id, feature_length;
  double feature_score;
  while ( (line = f_in.getline()) != NULL  ) {    

    //split
    line.trim("\n"); my_arr.init();
    my_arr.split(line, line_delimiter.cstr(), true);//["+1","1:0.01","100:0.86",...]

    // insert data
    feature_length = (unsigned int)(my_arr.length() - 1);// -1 is label, -1 1:0.1 2:0.4
    tarray_tstring my_arr2;
    tstring key, value;
    for (size_t j = 1; j < feature_length + 1;  j++) {
      my_arr2.init();
      my_arr2.split(my_arr[j], line_delimiter_between.cstr(), true);//"1:0.01"
      key.init(); key = my_arr2[0]; feature_id = (unsigned int)key.atoi();
      value.init(); value = my_arr2[1]; feature_score = value.atof();
      max_feature_id = get_max(&max_feature_id, &feature_id);
    }

  }

  return max_feature_id;
}

void 
load_data(RMatrixXf *X, RowVectorXf *y,
	  char *filename, 
	  tstring line_delimiter, tstring line_delimiter_between, tstring line_delimiter_within)
{
  stdstreamio f_in;
  tstring line;
  tarray_tstring my_arr;
  // file except
  if ( f_in.open("r", filename) < 0 ){
    fprintf(stderr, "Can not open %s\n", filename);
    exit(1);
  }

  unsigned int feature_id, feature_length;
  double feature_score;
  size_t i = 0;
  while ( (line = f_in.getline()) != NULL  ) {    

    //split
    line.trim("\n"); my_arr.init();
    my_arr.split(line, line_delimiter.cstr(), true);//["+1","1:0.01","100:0.86",...]

    // insert data
    size_t label =  my_arr[0].atoi();
    if (label > 0) {
      y->coeffRef(i) = 1;
    } else {
      y->coeffRef(i) = 0;
    }
    feature_length = (unsigned int)(my_arr.length() - 1);// -1 is label, -1 1:0.1 2:0.4
    tarray_tstring my_arr2;
    tstring key, value;
    for (size_t j = 1; j < feature_length + 1;  j++) {
      my_arr2.init();
      my_arr2.split(my_arr[j], line_delimiter_between.cstr(), true);//"1:0.01"
      key.init(); key = my_arr2[0]; feature_id = (unsigned int)key.atoi();
      value.init(); value = my_arr2[1]; feature_score = value.atof();
      X->coeffRef(i, feature_id - 1) = feature_score;
    }

    i += 1;
  }


}

void 
show_data_mat(RMatrixXf *X)
{
  printf("\n");
  for (size_t i = 0; i < X->rows(); i++)
    for (size_t j = 0; j < X->cols(); j++)
      fprintf(stdout, "%d:%d %f %p\n", i, j, X->coeffRef(i,j), &X->coeffRef(i,j));
}

void
show_data_vec(RowVectorXf *y)
{
  for (size_t i = 0; i < y->size(); i++)
    fprintf(stdout, "%d %f %p\n", i, y->coeffRef(i), &y->coeffRef(i));
}

