#include "timer.h"

namespace kmeans {

timer::timer() {
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
}

timer::~timer() {
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
}

void timer::start() {
    cudaEventRecord(m_start, 0);
}

float timer::stop() {
    float time;
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
    cudaEventElapsedTime(&time, m_start, m_stop);
    return time;
}

}

