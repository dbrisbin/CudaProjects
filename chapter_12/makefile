

NVCC = nvcc
CFLAGS = -Wall,-Werror,-Wextra
CUCOMPILEFLAGS = --device-c

CU_SRCS = merge_driver.cu merge.cu merge_utils.cu
CPP_SRCS = main.cpp
OBJS = $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)
EXECUTABLE=merge

DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXECUTABLE)
DBGOBJS = $(addprefix $(DBGDIR)/, $(OBJS))
DBGFLAGS = -g -G

RELDIR = release
RELEXE = $(RELDIR)/$(EXECUTABLE)
RELOBJS = $(addprefix $(RELDIR)/, $(OBJS))
RELFLAGS = -O3 -DNDEBUG

GENERATOR_SRC = generate_merge_data.cpp
GENERATOR_EXE = generate_merge_data

.PHONY: all clean debug prep release remake

debug: $(DBGEXE)

$(DBGEXE): 	$(DBGOBJS)	
	$(NVCC) --compiler-options $(CFLAGS) $(DBGFLAGS) -o $(DBGEXE) $^

$(DBGDIR)/%.o: %.cu
	$(NVCC) -c --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(DBGFLAGS) -o $@ $<

$(DBGDIR)/%.o: %.cpp
	$(NVCC) -c  --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(DBGFLAGS) -o $@ $<

release: $(RELEXE)

$(RELEXE): $(RELOBJS)
	$(NVCC) --compiler-options $(CFLAGS) $(RELFLAGS) -o $(RELEXE) $^

$(RELDIR)/%.o: %.cu
	$(NVCC) -c --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(RELFLAGS) -o $@ $^

$(RELDIR)/%.o: %.cpp
	$(NVCC) -c --compiler-options $(CFLAGS) $(CUCOMPILEFLAGS) $(RELFLAGS) -o $@ $^

generate_merge_data:
	$(NVCC) -o $(GENERATOR_EXE) $(GENERATOR_SRC)

prep:	
	@mkdir -p $(DBGDIR) $(RELDIR)

all: debug release generate_merge_data

clean:
	rm -f $(RELEXE) $(RELOBJS) $(DBGEXE) $(DBGOBJS) $(GENERATOR_EXE)

remake: clean all
