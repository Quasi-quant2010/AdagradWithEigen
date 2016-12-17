#include <stdio.h>
#include <iostream>

#include "src/util.h"
#include "src/arg_option.h"
#include "src/read_file.h"
#include "src/Adagrad.h"

int main(int argc, char **argv)
{

  // 1. option parser
  command_args *option_args = (command_args*)malloc(sizeof(command_args));
  printf("[Setting]\n");
  read_args(argc, argv, option_args);
  printf("\ttrain=%s,\n\ttest=%s,\n\tout_path=%s,\n\tstep_size=%1.2e,\n\tlambda=%1.2e,\n\tconvergence_rate=%1.2e,\n\tmax_iter=%d,\n\tmini_batch_size=%d\n",
         option_args->train_file, option_args->test_file, option_args->out_path,
         option_args->step_size, option_args->lambda, option_args->convergence_rate, option_args->max_iter, option_args->mini_batch_size);
  printf("[SGD Setting]\n");
  printf("\tepsilon=%1.2e,\n\tclip_threshold=%1.2e,\n\tclip_method=%s\n",
         option_args->epsilon, option_args->clip_threshold, option_args->clip_method);

  // 2. read train data and initialize feature weight vector
  tstring line_delimiter, line_delimiter_between, line_delimiter_within;
  line_delimiter.init(); line_delimiter.append(" ");
  line_delimiter_between.init(); line_delimiter_between.append(":");
  line_delimiter_within.init(); line_delimiter_within.append(":");

  // Train
  unsigned int feature_dim = 0;
  unsigned int train_size = 0;
  unsigned int *train_size_ptr = &train_size;
  feature_dim = get_feature_length(option_args->train_file,
				   line_delimiter, line_delimiter_between, line_delimiter_within);
  train_size = get_data_length(option_args->train_file);
  cout << "Train:(N,D)" << train_size << " :" << feature_dim << endl;
  RMatrixXf X_train = RMatrixXf::Zero(train_size,feature_dim);     // design matrix
  RowVectorXf y_train = RowVectorXf::Zero(train_size);             // label
  load_data(&X_train, &y_train, 
	    option_args->train_file, 
	    line_delimiter, line_delimiter_between, line_delimiter_within);
  // Test
  unsigned int test_size = 0;
  unsigned int *test_size_ptr = &test_size;
  test_size = get_data_length(option_args->test_file);
  RMatrixXf X_test = RMatrixXf::Zero(test_size,feature_dim);
  RowVectorXf y_test = RowVectorXf::Zero(test_size);
  load_data(&X_test, &y_test, 
	    option_args->test_file, 
	    line_delimiter, line_delimiter_between, line_delimiter_within);
  cout << "Train:" << train_size << " Test:" << test_size << endl;

  // 3. main
  Adagrad adglr = Adagrad(X_train.rows(), X_train.cols(),
			  option_args);

  const gsl_rng_type *T;
  gsl_rng *r;
  T = gsl_rng_mt19937;        // random generator
  r = gsl_rng_alloc(T);       // random gererator pointer
  gsl_rng_set(r, time(NULL)); // initialize seed for random generator by sys clock

  //  3.1 Train
  tstring dummy_fname;
  tstring out_fname;
  FILE *output;
  dummy_fname.init(); dummy_fname.assign("SGD");
  out_fname.init();
  out_fname = make_filename(dummy_fname, option_args);
  if ( (output = fopen(out_fname.cstr(), "w")) == NULL ) {
    printf("can not make output file");
    exit(1);
  }
  RowVectorXf w = RowVectorXf::Random(feature_dim); // feature weight. initialize
  double w_level = 0.0;                             // feature weight level
  double *w_level_ptr = &w_level;
  adglr.train(r, output,
	      &X_train, &y_train,
	      &w, w_level_ptr);
  fclose(output);


  //vector<double> adgscores;
  //adglr.predict(X_test, y_test, adgscores);
  //cout<<"=== Adagrad ==="<<endl;
  //cout <<"Accuracy:"<< adglr.Acc(adgscores, y_test) << endl;

  /*
  string fname = "adagrad_iters_";
  fname += option_args->clip_method;
  fname += ".txt";
  ofstream adgiofs(fname);
  for(int i=0; i < loss.size(); i++)
    adgiofs << loss[i] << endl;

 
  fname = "adagrad_weight_";
  fname += option_args->clip_method;
  fname += ".txt";
  ofstream adgwofs(fname);
  for(int i = 0; i< adglr.w.rows();i++)
    adgwofs << adglr.w(i) << endl;
  */
}
