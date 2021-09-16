
all:
	nvcc -G -odir build/ -dc -I include/ src/*.cu
	nvcc -o astar_gpu build/*.o

