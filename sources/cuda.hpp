#ifndef TERMSHADER_CUDA_HPP_


#include <cstdio>
#include <iostream>

#include "global.hpp"


#define CALL_CUDA(Func, ...) \
do {\
	cudaError_t ec = Func(__VA_ARGS__);\
	if (ec != cudaSuccess) {\
		std::cerr << "\033[31mFAILED\033[0m: " << cudaGetErrorString(ec) << std::endl;\
		return R(CudaError);\
	}\
}while(0)


#define APPEND_INFO(Los, Msg, ...) \
	do {\
		char buffer[256];\
		std::snprintf(buffer, ts::lengthof(buffer), Msg, ##__VA_ARGS__);\
		(Los) << "\033[34mINFO  \033[0m: " << buffer << std::endl;\
	}while(0)

#define APPEND_ERROR(Los, Msg, ...) \
	do {\
		char buffer[256];\
		std::snprintf(buffer, ts::lengthof(buffer), Msg, ##__VA_ARGS__);\
		(Los) << "\033[31mFAILED\033[0m: " << buffer << std::endl;\
	}while(0);



namespace ts {;


static Rcode setup_cuda(std::ostream& los) {

	int nbDevices;
	CALL_CUDA(cudaGetDeviceCount, &nbDevices);

	APPEND_INFO(los, "Number of CUDA devices: %d", nbDevices);

	if (0 == nbDevices) {
		APPEND_ERROR(los, "No CUDA devices!");
		return R(CudaSetup);
	}

	CALL_CUDA(cudaSetDevice, 0);

	APPEND_INFO(los, "Successfully set device 0");
	return R(Ok);
}


static Rcode teardown_cuda(std::ostream& los) {

	CALL_CUDA(cudaDeviceReset);
	APPEND_INFO(los, "Successfully reset device");
	return R(Ok);
}


template < typename T >
struct cuda_array_raii_t {
	T* arr;

    cuda_array_raii_t() noexcept : arr(nullptr) {}
	cuda_array_raii_t(T* a) noexcept : arr(a) {}
   ~cuda_array_raii_t() noexcept { if (arr != nullptr) cudaFree(arr); }

	void release(bool needFree=false) noexcept {
		if (needFree && !!arr) cudaFree(arr);
		arr = nullptr;
	}
};


}


#define TERMSHADER_CUDA_HPP_
#endif