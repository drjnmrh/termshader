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

#include <sys/time.h>
#include <sys/poll.h>
#include <unistd.h>
#include <termios.h>

#include <chrono>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <functional>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>

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
    f32 y = buf.h-r-1;
    shade_fire(pcell->bg, vec2_d(c, 2*y), T, vec2_d(buf.w, buf.h));
    shade_fire(pcell->fg, vec2_d(c, 2*y+1), T, vec2_d(buf.w, buf.h));
    pcell->c = (u32)U'▀';
}


static Rcode run_shaders(Display& display, std::ostream& los, std::atomic<bool>& stop) {

    color_buffer_f buf = display.buffer();
    buf.data = nullptr;

    u32 szbuf = sizeof(cell_t)*buf.h*buf.stride;

    CALL_CUDA(cudaMalloc, &buf.data, szbuf);
    cuda_array_raii_t<cell_t> gpuBufDataRaii(buf.data);

    static constexpr i32 cBlockSize = 32;

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


template <u32 S, typename T>
class MeasureSmoother {
public:
    explicit MeasureSmoother(T def) : _ix(0) {
        for (u32 i = 0; i < S; ++i) _ring[i] = def;
    }

    void addMeasurement(T v) noexcept {
        _ring[_ix] = v;
        _ix = (_ix + 1)%S;
    }

    T mean() const noexcept {
        T m = 0;
        for (u32 i = 0; i < S; ++i) m += _ring[i];
        return m / S;
    }

private:
    T _ring[S];
    u32 _ix;
};


struct Config {
    u32 width, height;
    i64 seed;
};


template < typename Container >
Rcode parse_arguments(int argc, char** argv, std::initializer_list<Container> opts) {

    for (std::size_t i = 1; i < argc; ++i) {
        for (const auto& p : opts) {
            std::string_view key = std::string_view(std::get<0>(p));
            if (std::strncmp(argv[i], key.data(), key.size()) == 0) {
                if (i+1 == argc) {
                    std::cout << "failed to read " << key << std::endl;
                    return R(InvalidInput);
                }

                char buffer[33];
                std::strncpy(buffer, argv[i+1], lengthof(buffer));
                Rcode rc = std::get<1>(p)(buffer);
                if (FAILED(rc)) {
                    std::cout << "failed to read " << key << std::endl;
                    return rc;
                }
                i += 1;
            }
        }
    }

    return R(Ok);
}


static char sStdoutBuffer[MB(10)];


int main(int argc, char** argv) {

    Config cfg{ .width = 128, .height = 64, .seed = -1 };
    using pr = std::pair<std::string_view, std::function<Rcode(const char*)> >;
    Rcode rc = parse_arguments(argc, argv, {
        pr{ "--seed"  , ([&](const char* b)->Rcode { cfg.seed = std::atoi(b); return R(Ok); }) },
        pr{ "--width" , ([&](const char* b)->Rcode { cfg.width = clamp(std::atoi(b), 16, 256); return R(Ok); }) },
        pr{ "--height", ([&](const char* b)->Rcode { cfg.height = clamp(std::atoi(b), 8, 128); return R(Ok); }) }
    });

    if (0 != setvbuf(stdout, sStdoutBuffer, _IOFBF, lengthof(sStdoutBuffer))) {
        std::cout << "FAILED TO SETUP STDOUT!" << std::endl;
        return -1;
    }

    std::stringstream los;

    if (cfg.seed < 0) {
        std::srand(std::time(0));
        APPEND_INFO(los, "SEED: time; RES: %dx%d", cfg.width, cfg.height);
    } else {
        std::srand(cfg.seed);
        APPEND_INFO(los, "SEED: %ld; RES: %dx%d", cfg.seed, cfg.width, cfg.height);
    }

    termios olds, news;
    tcgetattr(fileno(stdin), &olds);
    news = olds;
    news.c_lflag &= (~ICANON & ~ECHO);
    tcsetattr(fileno(stdin), TCSANOW, &news);
    fputs("\e[?25l", stdout);

    FWDI(setup_cuda, los);

    Display display;
    FWDI(display.setUp, cfg.width, cfg.height/2);

    std::atomic<bool> key(false);
    auto check_keypressed = [&]() { while (!keypressed()) std::this_thread::yield(); key.store(true, std::memory_order_release); };
    std::thread keychecker(check_keypressed);

    Rcode renderRc = R(Ok);
    auto run_render = [&]() { renderRc = run_shaders(display, los, key); };
    std::thread renderer(run_render);

    Stopwatch<float> sw, sw2;
    sw.reset(); sw2.reset();

    static constexpr f32 cFPS = 60.f;
    static constexpr f32 cFrameTime = 1.0f/cFPS;

    MeasureSmoother<16, f32> meanT(cFrameTime);

    FWDI(display.clear, stdout, los);
    while(!key.load(std::memory_order_acquire) && !FAILED(renderRc)) {
        float T = sw.measure();
        if (T > cFrameTime) {
            sw.reset();
            FWDI(display.output, stdout, los);

            meanT.addMeasurement(T);

            std::cout << los.str() << "FPS: " << 1.0f/meanT.mean() << std::endl;
            std::cout.flush();
        }
        std::this_thread::yield();
    }

    if (keychecker.joinable()) keychecker.join();
    if (renderer.joinable()) renderer.join();

    FWDI(teardown_cuda, std::cout);

    fputs("\e[?25h", stdout);
    tcsetattr(fileno(stdin), TCSANOW, &olds);

    tcflush(fileno(stdin), TCIOFLUSH);

    return 0;
}
