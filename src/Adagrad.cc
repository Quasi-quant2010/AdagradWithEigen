#include <math.h>

#include "read_file.h"
#include "arg_option.h"
#include "Adagrad.h"


Adagrad::Adagrad(size_t _N, size_t _D,
		 command_args* _option_args):
  N(_N), 
  D(_D),
  option_args(_option_args)
  /*
  E(RowVectorXf::Zero(_D+1))   // adaptive learning rate initialize,
                               // diagonal matrix. Therefore, VectorXd
                               //  feature-weight-dim + weight-level
                               //          _D         +      1
			       */
{}


void Adagrad::train(gsl_rng *r, FILE *fname, 
		    RMatrixXf *X, RowVectorXf *y,
		    RowVectorXf *w, double *w_level)
{
  // init
  double max_grad = 0.;
  if (this->option_args->clip_method == "euclidean_squeezing") {
    RMatrixXf G = RMatrixXf::Zero(this->N,this->D);
    for (size_t i=0; i < N; i++)
      for (size_t j=0; j < this->D; j++)
	G(i,j) = gsl_ran_flat(r, -0.5*float(this->D), 0.5*float(this->D));
  }

  SGD(r, fname,
      X, y, w, w_level);
  //Batch(fname, 
  //	X, y, w, w_level);
  /*
  if ( this->option_args->batch_sgd == 1) {
  } else if ( this->option_args->batch_sgd == 0) {
  }
  */
}


/* ---------------- Main ------------------------- */
void Adagrad::Batch(FILE *_fname,
		    RMatrixXf *_X, RowVectorXf *_y,
		    RowVectorXf *_w, double *_w_level)
{

  double learning_rate;
  double before_loss, after_loss, cur_loss_rate;
  unsigned int cur_iter;
  double *error = (double *)malloc(sizeof(double) * this->N);

  cur_loss_rate = this->option_args->convergence_rate + 1.0;
  before_loss = 1.0;
  cur_iter = 1;
  while (cur_loss_rate >= this->option_args->convergence_rate) {

    // 0. Set Learning Rate
    learning_rate = this->option_args->step_size / sqrt((double)cur_iter);

    // 1. inner_product
    init_vector(error, this->N);
    for (size_t i=0; i < this->N; i++) {
      double inner_product_ = inner_product(_X, i, _w);
      double pred = sigmoid(inner_product_ + *_w_level);
      error[i] = pred - (double)_y->coeffRef(i);
    }

    // 2. feature weight
    for (size_t j = 0; j < this->D; j++) {
      // calculate gradient
      double grad = 0.0;
      for (size_t i = 0; i < this->N; i++)
	grad += error[i] * _X->coeffRef(i,j);
      grad /= (double)this->N;
      // update
      _w->coeffRef(j) -= learning_rate * (grad + this->option_args->lambda * _w->coeffRef(j)); // L2-regularization
    }

    // 3. feature level
    // calculate gradient
    double grad = 0.0;
    for (size_t i = 0; i < this->N; i++)
      grad += error[i];
    grad /= (double)this->N;
    // update
    *_w_level -= learning_rate * (grad + this->option_args->lambda * *_w_level);              // L2-regularization

    // 4. Loss
    after_loss = LogLikelihood(_X, _y,
			       _w, _w_level);
    fprintf(_fname, "%d\t%f\n", cur_iter, after_loss);

    // 5. next iteration bool
    if (cur_iter == this->option_args->max_iter) break;
    before_loss = after_loss;
    cur_iter += 1;
  }

}

void Adagrad::SGD(gsl_rng* _r, FILE *_fname,
		  RMatrixXf *_X, RowVectorXf *_y,
		  RowVectorXf *_w, double *_w_level)
{
  //unsigned int i, j; //i is sample, j is feature index
  double learning_rate;
  double before_loss, after_loss, cur_loss_rate;
  unsigned int cur_iter;
  double inner_product_, predict_click;
  double *inner_product_ptr = &inner_product_;

  gsl_rng_type *T = (gsl_rng_type *)gsl_rng_mt19937; // random generator
  gsl_rng *r = gsl_rng_alloc(T);                     // random gererator pointer
  gsl_rng_set (r, time(NULL));                       // initialize seed for random generator by sys clock

  // stochastic gradient descent with mini batch
  //double *E = (double *)malloc(sizeof(double*) * (fw.size() + 1));
  //init_vector(E, fw.size() + 1);
  E_adaptive *E = (E_adaptive *)malloc(sizeof(E_adaptive) * (_w->size() + 1));
  init_struct_vector(E, _w->size() + 1);

  cur_loss_rate = this->option_args->convergence_rate + 1.0;
  before_loss = 1.0;
  cur_iter = 1;
  while (cur_loss_rate >= this->option_args->convergence_rate) {

    // 1. Sampling mini-batch data from train datas
    size_t *random_idx = (size_t *)malloc(sizeof(size_t) * this->option_args->mini_batch_size);
    for (size_t i = 0; i < this->option_args->mini_batch_size; i++)
      random_idx[i] = gsl_rng_uniform_int(r, this->N);

    // 2. calculate error
    double *error = (double *)malloc(sizeof(double) * this->option_args->mini_batch_size);
    init_vector(error, this->option_args->mini_batch_size);
    for (size_t i=0; i < this->option_args->mini_batch_size; i++) {
      inner_product_ = 0.0;
      double inner_product_ = inner_product(_X, random_idx[i], _w);
      double pred = sigmoid(inner_product_ + *_w_level);
      error[i] = pred - (double)_y->coeffRef(random_idx[i]);                                     // error
    }

    // 3. gradient descent
    //  3.1 feature weight
    for (size_t j=0; j < _w->size(); j++) {
      double grad = 0.0;
      // calculate gradient
      for (size_t i=0; i < this->option_args->mini_batch_size; i++)
	grad += _X->coeffRef(random_idx[i],j) * error[i];
      grad /= (double)this->option_args->mini_batch_size;                                        // gradient
      grad = get_adjust_gradient(this->option_args, 
				 E[j].max_grad, grad);                                           // Clipping

      // update
      E[j].E += pow(grad, 2.0);                                                                  // Cumulative (sub)gradient at iteration t for feature j
      E[j].max_grad = get_max(E[j].max_grad, grad);                                              // For Max Clipping
      learning_rate = get_learnig_rate(this->option_args, 
				       E[j].E, (double)cur_iter);                                // Adagrad : cumulative (sub)gradient at iteration t for feature j
      _w->coeffRef(j) -= learning_rate * (grad + this->option_args->lambda * _w->coeffRef(j));   // L2-regularization
    }
    //  3.2 feature level
    double grad=0.0;
    for (size_t i = 0; i < this->option_args->mini_batch_size; i++)
      grad += error[i];
    grad /= (double)this->option_args->mini_batch_size;                                          // gradient
    grad = get_adjust_gradient(this->option_args, 
			       E[_w->size()].max_grad, grad);                                    // Clipping
    E[_w->size()].E += pow(grad, 2.0);                                                           // Cumulative (sub)gradient at iteration t for feature j
    E[_w->size()].max_grad = get_max(E[_w->size()].max_grad, grad);                              // For Max Clipping
    learning_rate = get_learnig_rate(this->option_args,
				     E[_w->size()].E, (double)cur_iter);                         // Adagrad
    *_w_level -= learning_rate * (grad + this->option_args->lambda * *_w_level);                 // L2-regularization

    // 3. likelihood
    after_loss = LogLikelihood(_X, _y, _w, _w_level);
    fprintf(_fname, "%d\t%f\n", cur_iter, after_loss);

    // 4. next iteration bool
    if (cur_iter == this->option_args->max_iter) break;
    before_loss = after_loss;
    cur_iter += 1;
  }// over while  

}// over mini_batch_train


double Adagrad::get_max(double a, double b)
{
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

double Adagrad::get_min(double a, double b)
{
  if (a > b) {
    return b;
  } else {
    return a;
  }
}

double Adagrad::Clipping(double _grad, double _clip)
{
  _grad = get_max(get_min(_grad, _clip), -_clip);
  return _grad;
}

double Adagrad::MaxSqueezing(double _max_grad, double _grad, double _clip)
{
  if (_clip < _max_grad) {
    _grad *= 1.;
  } else {
    _grad *= _clip / _max_grad;
  }
  return _grad;
}

//void Adagrad::EuclideanSqueezing(double *grad)
//{
//}

double Adagrad::get_learnig_rate(const command_args* _option_args,
				 double _cumulative_gradient, double _cur_iter)
{
  tstring bool_clip; bool_clip.init(); bool_clip.append("clippng");
  tstring bool_max_squeeze; bool_max_squeeze.init(); bool_max_squeeze.append("max_squeezing");
  double learning_rate = 0.0;
  
  if (bool_clip.compare(_option_args->clip_method) == 0) {
    learning_rate = _option_args->step_size / sqrt( _cumulative_gradient + _option_args->epsilon);
  } else if (bool_max_squeeze.compare(_option_args->clip_method) == 0) {
    learning_rate = _option_args->step_size / sqrt( _cumulative_gradient + _option_args->epsilon);
  } else {
    learning_rate = _option_args->step_size / sqrt(_cur_iter);
  }
  return learning_rate;
}

double Adagrad::get_adjust_gradient(const command_args* _option_args,
				    double _max_grad, double _grad)
{
  tstring bool_clip; bool_clip.init(); bool_clip.append("clippng");
  tstring bool_max_squeeze; bool_max_squeeze.init(); bool_max_squeeze.append("max_squeezing");

  double grad_ = _grad;
  if (bool_clip.compare(_option_args->clip_method) == 0) {
    // clipping
    grad_ = Clipping(_grad, _option_args->clip_threshold);
  } else if (bool_max_squeeze.compare(_option_args->clip_method) == 0) {
    // Max Squeezing
    grad_ = MaxSqueezing(_max_grad, _grad, _option_args->clip_threshold);
  }

  return grad_;
}
 
 
 
/* ---------------- Loss Function ------------------- */
double Adagrad::LogLikelihood(const RMatrixXf *__X, const RowVectorXf *__y,
			      const RowVectorXf *__w, const double *__w_level)
{
  double loss = 0.0;
  for (size_t i = 0; i < this->N; i++) {

    double inner_product_ = inner_product(__X, i, __w);
    double pred = sigmoid(inner_product_ + *__w_level);
    if (__y->coeffRef(i) == 1) {
      loss += log(pred);
    } else if (__y->coeffRef(i) == 0) {
      loss += log(1.0 - pred);
    }
  }

  return -loss / (double)this->N +					\
    0.5 * this->option_args->lambda * (__w->squaredNorm() +		\
				       sqrt(pow(*__w_level,2.0)));
  
}


/* ---------------- Elementary Function ------------------- */
double Adagrad::inner_product(const RMatrixXf *__X, size_t i, const RowVectorXf *__w){
  float tmp = __X->row(i).dot(*__w);
  return tmp;
}

double Adagrad::sigmoid(double z)
{
  if (z >  6.) {
    return 1.0;
  } else if (z < -6.) {
    return 0.0;
  } else {
    return 1.0 / (1 + exp(-z));
  }
 }

void Adagrad::init_vector(double* a, size_t len)
{
  for (size_t i=0; i < len; i++)
    a[i] = 0.0;
}

void Adagrad::init_struct_vector(E_adaptive* _E, size_t len)
{
  for (size_t i=0; i < len; i++) {
    _E[i].E = 0.0;
    _E[i].max_grad = 0.0;
  }
}


/*
double Adagrad::Acc(vector<double>& pred,VectorXd &l){
  int t =0;
  double loss =0;
  for(int i = 0; i< pred.size();i++){
    loss += (l(i) - pred[i]) * (l(i) - pred[i]);
    int s = pred[i] > 0.5 ? 1 : 0;
    if(s == l(i)){
      t++;
    }
  }
  //return (double)t/pred.size();
  return loss / pred.size();
}

void Adagrad::predict(MatrixXd& _x,VectorXd& _l,vector<double>& ret){
  for(int i = 0; i < _x.rows(); i++){
    double inner_product_ = inner_product(_x, i);
    double pred = sigmoid(inner_product_ + this->w_level);
    ret.push_back(pred);
  }
}
*/
