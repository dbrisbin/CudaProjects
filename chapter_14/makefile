

NVCC = nvcc
CFLAGS = -Wall,-Werror,-Wextra
CUCOMPILEFLAGS = --device-c

CU_SRCS = spmv_driver.cu spmv.cu spmv_utils.cu
CPP_SRCS = main.cpp
OBJS = $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)
EXECUTABLE=spmv

DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXECUTABLE)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJS))
DBGFLAGS = -g -G

RELDIR = release
RELEXE = $(RELDIR)/$(EXECUTABLE)
RELOBJS = $(addprefix $(RELDIR)/, $(OBJS))
RELFLAGS = -O3 -DNDEBUG

GENERATOR_SRC = generate_sparse_matrix_data.cpp
GENERATOR_EXE = generate_sparse_matrix_data

DOC_CONFIG = doxygen.conf
DOC_DIRS = html latex

.PHONY: all clean debug release remake documentation generate_sparse_matrix_data

debug: $(DBGEXE)

$(DBGEXE):	$(DBGOBJS)
	@mkdir -p $(DBGDIR)
	$(NVCC) --compiler-options $(CFLAGS) $(DBGFLAGS) -o $(DBGEXE) $^


$(DBGDIR)/%.o: %.cu
	@mkdir -p $(DBGDIR)
	$(NVCC) -c --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(DBGFLAGS) -o $@ $<

$(DBGDIR)/%.o: %.cpp
	@mkdir -p $(DBGDIR)
	$(NVCC) -c  --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(DBGFLAGS) -o $@ $<

release: $(RELEXE)

$(RELEXE): $(RELOBJS)
	@mkdir -p $(RELDIR)
	$(NVCC) --compiler-options $(CFLAGS) $(RELFLAGS) -o $(RELEXE) $^

$(RELDIR)/%.o: %.cu
	@mkdir -p $(RELDIR)
	$(NVCC) -c --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(RELFLAGS) -o $@ $^

$(RELDIR)/%.o: %.cpp
	@mkdir -p $(RELDIR)
	$(NVCC) -c --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(RELFLAGS) -o $@ $^

generate_sparse_matrix_data:
	$(NVCC) -o $(GENERATOR_EXE) $(GENERATOR_SRC)

convert_coo_to_csr: debug/spmv_utils.o debug/convert_coo_to_csr_driver.o
		$(NVCC) --compiler-options $(CFLAGS) $(DBGFLAGS) -o convert_coo_to_csr $^

documentation:
	doxygen $(DOC_CONFIG)

all: debug release generate_sparse_matrix_data documentation

clean:
	rm -f $(RELEXE) $(RELOBJS) $(DBGEXE) $(DBGOBJS) $(GENERATOR_EXE) convert_coo_to_csr
	rm -fr $(DOC_DIRS) $(DBGDIR) $(RELDIR)

remake: clean all
