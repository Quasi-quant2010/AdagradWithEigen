#ifndef ADAGRAD_H
#define ADAGRAD_H

#include "util.h"
#include "arg_option.h"
#include "read_file.h"

class Adagrad{
private:
  int N;                        // the num of samples
  int D;                        // the num of features
  //RowVectorXf E;                // gradient for adaptive learning rate


  /* ---------------- Main ------------------------- */
  void Batch(FILE* _fname,
	     RMatrixXf* _X, RowVectorXf* _y,
	     RowVectorXf* _w, double* _w_level);
  void SGD(gsl_rng* _r, FILE* _fname,
	   RMatrixXf* _X, RowVectorXf* _y,
	   RowVectorXf* _w, double* _w_level);

  /* ---------------- Adagrad ------------------------- */
  double get_learnig_rate(const command_args* _option_args, 
			  double _cumulative_gradient, double _cur_iter);
  double get_adjust_gradient(const command_args* _option_args,
			     double _max_grad, double _grad);
  double get_max(double a, double b);
  double get_min(double a, double b);
  double Clipping(double _grad, double _clip);
  double MaxSqueezing(double _max_grad, double _grad, double _clip);

  /* ---------------- Loss Function ------------------- */
  double LogLikelihood(const RMatrixXf *__X, const RowVectorXf *__y,
		       const RowVectorXf *__w, const double *__w_level);

  /* ---------------- Elementary Function ------------------- */
  void init_vector(double* a, size_t len);
  void init_struct_vector(E_adaptive* a, size_t len);
  double inner_product(const RMatrixXf *__X, size_t i, const RowVectorXf *__w);
  double sigmoid(double z);

public:
  command_args* option_args;     // option_parser's args
  /* <- option_argsの中身
   double lambda;                // regularization
   double eta;                   // learning rate
   double epsilon;               // adagrad's epsilon
   double clip;                  // threshold in clipping
   string clip_method;           // clipping method, clipping, squeese, euclidian
   unsigned int iteration;       // max iteration
   unsigned int mini_batch;      // mini_batch size
   double convergence_threshold; // early stopping
   unsigned int batch_sgd;       // if 1, optimization is SGD
  */

  Adagrad(size_t, size_t,
	  command_args*);

  void train(gsl_rng*, FILE*,
	     RMatrixXf*, RowVectorXf*,
	     RowVectorXf*, double*);
  /*
    _r : 
    _fnamae :
    _batch_sgd : If _batch_sgd=1, then SGD
    _X : designe matrix
    _y : label
    _w : feature weight
    _w_level : feature weight level
   */
  //void predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret);

  ~Adagrad()
    {
      // do nothing
    }
  
};

#endif
