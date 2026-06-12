#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
int cuda_memcpy_to_device(void* dest, const void* src, size_t size);
int cuda_memcpy_to_host(void* dest, const void* src, size_t size);

float calculate_motion_magnitude(const unsigned char* prev_host, const unsigned char* curr_host, int width, int height);

#ifdef __cplusplus
}
#endif

#endif // OPTICAL_FLOW_H
