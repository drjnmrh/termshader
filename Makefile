build/termshader: sources/main.cu
	mkdir -p ${dir $@} 
	nvcc -std=c++17 -O3 -Isources sources/main.cu -o build/termshader

clean:
	rm -rf build
