

NVCC = nvcc
CFLAGS = -Wall,-Werror,-Wextra

CU_SRCS = stencilDriver.cu stencil3d.cu 
C_SRCS = main.c matrixUtils.c
C_OBJS = $(C_SRCS:.c=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)
EXECUTABLE=stencil3d

DBGDIR = debug
DBGEXE = $(DBGDIR)/$(EXECUTABLE)
DBGCOBJS = $(addprefix $(DBGDIR)/, $(C_OBJS))
DBGCUOBJS = $(addprefix $(DBGDIR)/, $(CU_OBJS))
DBGCFLAGS = -g,-O0,-DDEBUG
DBGCUFLAGS = -g -G

RELDIR = release
RELEXE = $(RELDIR)/$(EXECUTABLE)
RELCOBJS = $(addprefix $(RELDIR)/, $(C_OBJS))
RELCUOBJS = $(addprefix $(RELDIR)/, $(CU_OBJS))
RELCFLAGS = -O3 -DNDEBUG
RELCUFLAGS = -O3 -DNDEBUG

.PHONY: all clean debug prep release remake

debug: $(DBGEXE)

$(DBGEXE): 	$(DBGCOBJS) $(DBGCUOBJS)	
	$(NVCC) --compiler-options $(DBGCFLAGS) --compiler-options $(CFLAGS) $(DBGCUFLAGS) -o $(DBGEXE) $^

$(DBGDIR)/%.o: %.cu
	$(NVCC) -c --compiler-options $(DBGCFLAGS) --compiler-options $(CFLAGS) $(DBGCUFLAGS) -o $@ $<

$(DBGDIR)/%.o: %.c
	$(NVCC) -c --compiler-options $(DBGCFLAGS) --compiler-options $(CFLAGS) $(DBGCUFLAGS) -o $@ $<

release: $(RELEXE)

$(RELEXE): $(RELCOBJS) $(RELCUOBJS)
	$(NVCC) --compiler-options $(RELCFLAGS) --compiler-options $(CFLAGS) $(RELCUFLAGS) -o $(RELEXE) $^

$(RELDIR)/%.o: %.c
	$(NVCC) -c --compiler-options $(RELCFLAGS) --compiler-options $(CFLAGS) $(RELCUFLAGS) -o $@ $^

$(RELDIR)/%.o: %.cu
	$(NVCC) -c --compiler-options $(RELCFLAGS) --compiler-options $(CFLAGS) $(RELCUFLAGS) -o $@ $^
prep:	
	@mkdir -p $(DBGDIR) $(RELDIR)

remake: clean all

clean:
	rm -f $(RELEXE) $(RELCOBJS) $(RELCUOBJS) $(DBGEXE) $(DBGCOBJS) $(DBGCUOBJS)