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

/* Priority 1+2: Async stream + pinned memory + NPP norm-diff. */
float calculate_motion_magnitude(const unsigned char* prev_host, const unsigned char* curr_host, int width, int height);

/* Priority 3: GPU-side YOLO preprocessing.
 * Converts a src_w x src_h BGR uint8 frame (HWC, host) into a
 * 640 x 640 RGB float32 CHW tensor normalised to [0,1].
 * Writes 3*640*640 floats into out_chw_host (caller-allocated).
 */
void preprocess_frame_for_yolo(const unsigned char* bgr_host, int src_w, int src_h, int src_step, float* out_chw_host);

#ifdef __cplusplus
}
#endif

#endif // OPTICAL_FLOW_H
