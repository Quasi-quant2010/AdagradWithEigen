# Required Libaray
- Ubuntu
  - 12.04.5 LTS
- GSL
  - The GNU Scientific Library (GSL) is a numerical library for C and C++ programmers
  - version 1.16
  - https://www.gnu.org/software/gsl/
- Eigen
  - https://bitbucket.org/eigen/eigen
- SLLIB
  - Very useful libraries handling various streams, string and FITS I/O for C users
  - version 1.4.2
  - http://www.ir.isas.jaxa.jp/~cyamauch/sli/index.ni.html

# example data
- https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/australian_scale

# Sample Command
- make
- ./main --train_file sample/train.txt \
         --test_file sample/test.txt \
		 --step_size 1.0 \
		 --epsilon 1.0e-8 \
		 --clip_threshold 1.0 \
		 --clip_method clipping \
		 --lambda 0.1 \
		 --convergence_rate 0.1 \
		 --max_iter 100 \
		 --mini_batch_size 5 \
		 --out_path ./ \
		 --out_fname L2		 

# Sample Result
- in sample folder, there is a Batch VS SGD LogLoss Plot.
