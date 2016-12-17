CXX = g++
CXXFLAGS = -std=c++0x -Wall -O2 -march=native -funroll-loops -fopenmp -pthread -I/usr/local/eigen -I/usr/local/gsl/include -I/usr/local/sllib/include
LDFLAGS = -L/usr/local/gsl/lib -L/usr/local/sllib/lib64
LDLIBS = -lgsl -lopenblas -lm -lsllib
TARGET = main
SRCS = main.cc src/util.cc src/arg_option.cc src/read_file.cc src/Adagrad.cc
OBJS = $(SRCS:.cc=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
