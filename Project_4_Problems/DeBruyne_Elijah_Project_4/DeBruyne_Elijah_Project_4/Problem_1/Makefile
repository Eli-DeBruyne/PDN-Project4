NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	    = gpu_mining_problem1  # Changed from gpu_mining_starter to gpu_mining_problem1
OBJ	    = gpu_mining_problem1.o support.o  # Changed from gpu_mining_starter.o to gpu_mining_problem1.o

default: $(EXE)

$(EXE):$(OBJ)
	$(NVCC) -o $@ $^ $(LD_FLAGS)

# ------------------------------------------------------------------------------- #

# input file and size
FILE     = debug_1k.csv
SIZE	 = 1000

# number of trials to run
TRIALS_A = 5000
TRIALS_B = 10000

# output file suffix
OUT_A    = 1k_5k
OUT_B    = 1k_10k


run:
	clear 
	make
	./gpu_mining_problem1 $(FILE) $(SIZE) $(TRIALS_A) _out_$(OUT_A).csv _time_$(OUT_A).csv  # Changed from ./gpu_mining_starter to ./gpu_mining_problem1
	./gpu_mining_problem1 $(FILE) $(SIZE) $(TRIALS_B) _out_$(OUT_B).csv _time_$(OUT_B).csv  # Changed from ./gpu_mining_starter to ./gpu_mining_problem1


# ------------------------------------------------------------------------------- #

gpu_mining_problem1.o: gpu_mining_problem1.cu nonce_kernel.cu hash_kernel.cu support.h  # Changed from gpu_mining_starter.cu to gpu_mining_problem1.cu
	$(NVCC) -c -o $@ gpu_mining_problem1.cu $(NVCC_FLAGS) $(LD_FLAGS)  # Changed from gpu_mining_starter.cu to gpu_mining_problem1.cu

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
