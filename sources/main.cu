#include <sys/time.h>
#include <sys/poll.h>
#include <unistd.h>
#include <termios.h>

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <locale>
#include <codecvt>

#include "cuda.hpp"
#include "display.hpp"
#include "global.hpp"
#include "maths.hpp"

#include "display.cpp"

#include "maths.cu"
#include "fire.cu"


using namespace ts;


__global__ static void do_shading(color_buffer_f buf, float T) {

	u32 c = threadIdx.x + blockIdx.x * blockDim.x;
	u32 r = threadIdx.y + blockIdx.y * blockDim.y;

	if (r >= buf.h || c >= buf.w) return;

	cell_t* pcell = &buf.data[r*buf.stride + c];
	shade_fire(pcell->bg, vec2_d(c, buf.h-r-1), T, vec2_d(buf.w, buf.h));
	pcell->c = (u32)U' ';
}


static Rcode run_shaders(Display& display, std::ostream& los, std::atomic<bool>& stop) {

	color_buffer_f buf = display.buffer();
	buf.data = nullptr;

	u32 szbuf = sizeof(cell_t)*buf.h*buf.stride;
	
    CALL_CUDA(cudaMalloc, &buf.data, szbuf);
    cuda_array_raii_t<cell_t> gpuBufDataRaii(buf.data);

	static constexpr i32 cBlockSize = 16;

	vec<2, i32> nbblocks = ceil(vec2((f32)buf.w, (f32)buf.h)/cBlockSize);

	APPEND_INFO(los, "nbblocks: %d, %d", nbblocks.x, nbblocks.y);

	dim3 szblock(cBlockSize, cBlockSize);
	dim3 szgrid(nbblocks.x, nbblocks.y);

	Stopwatch<float> swtotal;
	swtotal.reset();
	while(!stop.load(std::memory_order_acquire)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));

		do_shading<<<szgrid, szblock>>>(buf, swtotal.measure());

		CALL_CUDA(cudaPeekAtLastError);
		CALL_CUDA(cudaDeviceSynchronize);


        CALL_CUDA(cudaMemcpy, display.buffer().data, buf.data, szbuf, cudaMemcpyDeviceToHost);

		display.flip();
	}

	return R(Ok);
}


static bool keypressed() {

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(fileno(stdin), &fds);

	timeval tout;
	tout.tv_sec = 0;
	tout.tv_usec = 100;

	int r = select(fileno(stdin)+1, &fds, NULL, NULL, &tout);
	return r > 0;
}


static f32 select_fps(f32 time, f32 mean, f32 freq, f32 amp, f32 rnd) {
	return mean + amp*std::sin((time*freq + rnd*((std::rand()%10000)/10000.0f - 0.5f))*2.f*M_PI);
}


int main(int argc, char** argv) {

	std::srand(std::time(0));

	termios olds, news;
	tcgetattr(fileno(stdin), &olds);
	news = olds;
	news.c_lflag &= (~ICANON & ~ECHO);
	tcsetattr(fileno(stdin), TCSANOW, &news);
	fputs("\e[?25l", stdout);

	std::stringstream los;
	FWDI(setup_cuda, los); 

	Display display;
	FWDI(display.setUp, 128, 32);

	std::atomic<bool> key(false);
	auto check_keypressed = [&]() { while (!keypressed()) std::this_thread::yield(); key.store(true, std::memory_order_release); };
	std::thread keychecker(check_keypressed);

	Rcode renderRc = R(Ok);
	auto run_render = [&]() { renderRc = run_shaders(display, los, key); };
	std::thread renderer(run_render);

	Stopwatch<float> sw, sw2;
	sw.reset(); sw2.reset();

	static constexpr f32 cFPS = 30.0f;

	float ring[16];
	for(u32 i = 0; i < lengthof(ring); ++i) ring[i] = 1.0f/cFPS;
	u32 ixring=0;
	f32 fps = cFPS;

	while(!key.load(std::memory_order_acquire) && !FAILED(renderRc)) {
		float T = sw.measure();
		if (T > 1.0f/fps) {
			fps = select_fps(sw2.measure(), cFPS, 0.2f, 0.05f, 1.5f);
			sw.reset();
			display.output(std::cout);
			ring[ixring] = T;
			ixring = (ixring+1)%lengthof(ring);
			f32 meanT = 0.0f;
			for (u32 i = 0; i < lengthof(ring); ++i) meanT += ring[i];
			meanT /= lengthof(ring);
			std::cout << los.str() << "FPS: " << 1.0f/meanT << std::endl;
			std::cout.flush();
		}
		std::this_thread::yield();
	}

	if (keychecker.joinable()) keychecker.join();
	if (renderer.joinable()) renderer.join();

	FWDI(teardown_cuda, std::cout);

	fputs("\e[?25h", stdout);
	tcsetattr(fileno(stdin), TCSANOW, &olds);

	return 0;
}
