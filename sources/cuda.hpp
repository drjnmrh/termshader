/**
 * MIT License
 *
 * Copyright (c) 2024 Stoned Fox
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef TERMSHADER_CUDA_HPP_


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
