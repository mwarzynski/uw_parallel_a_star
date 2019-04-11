
all:
	nvcc -odir build/ -c -I include/ src/*.cu
	nvcc -o astar_gpu build/*.o

