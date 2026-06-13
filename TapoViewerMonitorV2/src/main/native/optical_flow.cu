/*
 * optical_flow.cu  —  GPU motion & preprocessing utilities
 *
 * Priority 1: Dedicated per-thread CUDA stream + cudaMallocHost pinned buffers
 *             + cudaMemcpyAsync so the motion path never blocks the ONNX stream.
 * Priority 2: nppiNormDiff_L1_8u_C1R_Ctx replaces the hand-rolled abs-diff kernel.
 *             cudaStreamSynchronize(stream) instead of cudaDeviceSynchronize().
 * Priority 3: preprocess_frame_for_yolo() — full GPU preprocessing pipeline for
 *             YOLOv8-pose input: resize → BGR→RGB → uint8→float32 → /255 → HWC→CHW.
 */

#include "optical_flow.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

/* NPP umbrella headers */
#include <npp.h>
#include <nppi.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] %s failed in %s:%d - %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

#define NPP_CHECK(call) \
    do { \
        NppStatus status = (call); \
        if (status != NPP_SUCCESS) { \
            fprintf(stderr, "[NPP ERROR] %s failed in %s:%d - status %d\n", \
                    #call, __FILE__, __LINE__, (int)status); \
        } \
    } while (0)

/* ========================================================================
 * HWC → CHW float32 rearrangement kernel
 * Converts interleaved [H×W×C] float layout to planar [C×H×W].
 * One thread per output element.  Launched on the YOLO stream.
 * ======================================================================== */
__global__ void hwc_to_chw_kernel(const float* __restrict__ hwc,
                                   float* __restrict__ chw,
                                   int spatial,   /* H*W           */
                                   int channels)  /* C (always 3)  */
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= spatial * channels) return;

    int c  = tid / spatial;          /* which channel (0=R,1=G,2=B) */
    int sp = tid % spatial;          /* spatial index (row*W + col)  */
    /* hwc[sp*C + c] → chw[c*spatial + sp]  (= chw[tid]) */
    chw[tid] = hwc[sp * channels + c];
}

/* ========================================================================
 * Legacy helpers (kept for CudaBridge backward-compat)
 * ======================================================================== */
extern "C" {

void* cuda_malloc(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

int cuda_memcpy_to_device(void* dest, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyH2D failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_memcpy_to_host(void* dest, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyD2H failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

/* ========================================================================
 * calculate_motion_magnitude  — Priority 1 + 2
 * ======================================================================== */
float calculate_motion_magnitude(const unsigned char* prev_host,
                                  const unsigned char* curr_host,
                                  int width, int height)
{
    int size = width * height;
    if (size <= 0) return 0.0f;

    /* Per-thread persistent state ---------------------------------------- */
    thread_local static cudaStream_t stream       = nullptr;
    thread_local static unsigned char* d_prev     = nullptr;
    thread_local static unsigned char* d_curr     = nullptr;
    thread_local static Npp64f*        d_normDiff = nullptr; /* device scalar result */
    thread_local static Npp64f*        h_result   = nullptr; /* pinned host scalar   */
    thread_local static Npp8u*         d_scratch  = nullptr;
    thread_local static unsigned char* h_prev_pin = nullptr;
    thread_local static unsigned char* h_curr_pin = nullptr;
    thread_local static int cached_w = 0, cached_h = 0;

    /* One-time per-thread initialisation ---------------------------------- */
    if (!stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        if (!stream) return 0.0f;
        CUDA_CHECK(cudaMallocHost((void**)&h_result, sizeof(Npp64f)));
        CUDA_CHECK(cudaMalloc((void**)&d_normDiff, sizeof(Npp64f)));
        if (!h_result || !d_normDiff) return 0.0f;
    }

    /* Set up thread-local NPP Stream Context */
    NppStreamContext nppCtx;
    NPP_CHECK(nppGetStreamContext(&nppCtx));
    nppCtx.hStream = stream;

    /* Reallocate buffers whenever bounding-box dimensions change ---------- */
    if (width != cached_w || height != cached_h) {
        /* Free previous allocations */
        if (h_prev_pin) { cudaFreeHost(h_prev_pin); h_prev_pin = nullptr; }
        if (h_curr_pin) { cudaFreeHost(h_curr_pin); h_curr_pin = nullptr; }
        if (d_prev)     { cudaFree(d_prev);          d_prev     = nullptr; }
        if (d_curr)     { cudaFree(d_curr);          d_curr     = nullptr; }
        if (d_scratch)  { cudaFree(d_scratch);       d_scratch  = nullptr; }

        /* Pinned host buffers for zero-copy async transfer */
        CUDA_CHECK(cudaMallocHost((void**)&h_prev_pin, size));
        CUDA_CHECK(cudaMallocHost((void**)&h_curr_pin, size));

        /* Device image buffers */
        CUDA_CHECK(cudaMalloc((void**)&d_prev, size));
        CUDA_CHECK(cudaMalloc((void**)&d_curr, size));

        if (!h_prev_pin || !h_curr_pin || !d_prev || !d_curr) {
            fprintf(stderr, "[CUDA ERROR] Failed to allocate buffers in %s\n", __FUNCTION__);
            return 0.0f;
        }

        /* NPP scratch buffer (query required size for this ROI) */
        int scratchBytes = 0;
        NppiSize roi = {width, height};
        NPP_CHECK(nppiNormDiffL1GetBufferHostSize_8u_C1R_Ctx(roi, &scratchBytes, nppCtx));
        if (scratchBytes > 0) {
            CUDA_CHECK(cudaMalloc((void**)&d_scratch, scratchBytes));
            if (!d_scratch) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate scratch in %s\n", __FUNCTION__);
                return 0.0f;
            }
        }

        cached_w = width;
        cached_h = height;
    }

    /* Stage pageable host data into pinned buffers (CPU memcpy, negligible) */
    memcpy(h_prev_pin, prev_host, size);
    memcpy(h_curr_pin, curr_host, size);

    /* Async H→D transfers on our dedicated stream ------------------------- */
    CUDA_CHECK(cudaMemcpyAsync(d_prev, h_prev_pin, size, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_curr, h_curr_pin, size, cudaMemcpyHostToDevice, stream));

    /* NPP L1 norm-diff (sum |prev[i] - curr[i]|) on the same stream. */
    NppiSize roi = {width, height};
    NPP_CHECK(nppiNormDiff_L1_8u_C1R_Ctx(
        (const Npp8u*)d_prev, width,
        (const Npp8u*)d_curr, width,
        roi, d_normDiff, d_scratch, nppCtx));

    /* Async D→H for the single scalar result into pinned host memory ------ */
    CUDA_CHECK(cudaMemcpyAsync(h_result, d_normDiff, sizeof(Npp64f),
                    cudaMemcpyDeviceToHost, stream));

    /* Sync ONLY this stream — ONNX inference stream is not affected -------- */
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaStreamSynchronize failed in %s: %s\n",
                __FUNCTION__, cudaGetErrorString(syncErr));
        return 0.0f;
    }

    return (float)(*h_result / size);
}

/* ========================================================================
 * preprocess_frame_for_yolo  — Priority 3
 * ======================================================================== */
void preprocess_frame_for_yolo(const unsigned char* bgr_host,
                                int src_w, int src_h,
                                int src_step,
                                float* out_chw_host)
{
    static const int DST_W  = 640;
    static const int DST_H  = 640;
    static const int DST_SP = DST_W * DST_H;        /* spatial pixels */
    static const int DST_C  = 3;                    /* channels       */

    thread_local static cudaStream_t stream       = nullptr;
    /* Variable-size input buffer (resizes with source frame) */
    thread_local static unsigned char* d_bgr_src  = nullptr;
    thread_local static int cached_src_w = 0, cached_src_h = 0;
    /* Fixed-size 640×640 working buffers (allocated once) */
    thread_local static unsigned char* d_bgr_640  = nullptr;
    thread_local static unsigned char* d_rgb_640  = nullptr;
    thread_local static float*         d_hwc_f32  = nullptr;
    thread_local static float*         d_chw_f32  = nullptr;
    thread_local static bool           fixed_ready = false;

    /* One-time stream creation ------------------------------------------- */
    if (!stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        if (!stream) return;
    }

    /* Set up thread-local NPP Stream Context */
    NppStreamContext nppCtx;
    NPP_CHECK(nppGetStreamContext(&nppCtx));
    nppCtx.hStream = stream;

    /* Allocate fixed-size buffers once per thread ------------------------- */
    if (!fixed_ready) {
        size_t u8_sz  = (size_t)DST_W * DST_H * DST_C;
        size_t f32_sz = (size_t)DST_W * DST_H * DST_C * sizeof(float);
        CUDA_CHECK(cudaMalloc((void**)&d_bgr_640, u8_sz));
        CUDA_CHECK(cudaMalloc((void**)&d_rgb_640, u8_sz));
        CUDA_CHECK(cudaMalloc((void**)&d_hwc_f32, f32_sz));
        CUDA_CHECK(cudaMalloc((void**)&d_chw_f32, f32_sz));
        if (!d_bgr_640 || !d_rgb_640 || !d_hwc_f32 || !d_chw_f32) {
            fprintf(stderr, "[CUDA ERROR] Failed to allocate fixed buffers in %s\n", __FUNCTION__);
            return;
        }
        fixed_ready = true;
    }

    /* Resize source-frame device buffer if resolution changed ------------- */
    if (src_w != cached_src_w || src_h != cached_src_h) {
        if (d_bgr_src) { CUDA_CHECK(cudaFree(d_bgr_src)); d_bgr_src = nullptr; }
        size_t input_sz = (size_t)src_w * src_h * DST_C;
        CUDA_CHECK(cudaMalloc((void**)&d_bgr_src, input_sz));
        if (!d_bgr_src) {
            fprintf(stderr, "[CUDA ERROR] Failed to allocate input buffer of size %zu in %s\n", input_sz, __FUNCTION__);
            return;
        }
        cached_src_w = src_w;
        cached_src_h = src_h;
    }

    /* 1. Upload source BGR frame via robust 2D copy (handles padding) ----- */
    CUDA_CHECK(cudaMemcpy2DAsync(d_bgr_src, src_w * DST_C,
                                 bgr_host, src_step,
                                 src_w * DST_C, src_h,
                                 cudaMemcpyHostToDevice, stream));

    NppiSize srcSize = {src_w, src_h};
    NppiRect srcROI  = {0, 0, src_w, src_h};
    NppiSize dstSize = {DST_W, DST_H};
    NppiRect dstROI  = {0, 0, DST_W, DST_H};

    /* 2. Resize BGR: src_w×src_h → 640×640 -------------------------------- */
    NPP_CHECK(nppiResize_8u_C3R_Ctx(
        (const Npp8u*)d_bgr_src, src_w * DST_C, srcSize, srcROI,
        (Npp8u*)d_bgr_640,       DST_W * DST_C, dstSize, dstROI,
        NPPI_INTER_LINEAR, nppCtx));

    /* 3. BGR → RGB channel reorder (aDstOrder maps output[i] = src[order[i]]) */
    const int rgbOrder[3] = {2, 1, 0};
    NPP_CHECK(nppiSwapChannels_8u_C3R_Ctx(
        (const Npp8u*)d_bgr_640, DST_W * DST_C,
        (Npp8u*)d_rgb_640,       DST_W * DST_C,
        dstSize, rgbOrder, nppCtx));

    /* 4. uint8 → float32 (still HWC, interleaved) ------------------------- */
    NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(
        (const Npp8u*)d_rgb_640,  DST_W * DST_C,
        (Npp32f*)d_hwc_f32,       DST_W * DST_C * sizeof(float),
        dstSize, nppCtx));

    /* 5. Normalise /255 in-place (HWC float) ------------------------------ */
    const Npp32f kNorm[3] = {255.0f, 255.0f, 255.0f};
    NPP_CHECK(nppiDivC_32f_C3IR_Ctx(kNorm, (Npp32f*)d_hwc_f32,
                      DST_W * DST_C * sizeof(float), dstSize, nppCtx));

    /* 6. HWC → CHW rearrangement via custom kernel on the same stream ----- */
    int total   = DST_SP * DST_C;          /* 640*640*3 = 1,228,800 */
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    hwc_to_chw_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)d_hwc_f32, (float*)d_chw_f32, DST_SP, DST_C);
    CUDA_CHECK(cudaGetLastError());

    /* 7. Async D→H into caller's buffer (direct ByteBuffer from Java) ----- */
    size_t out_bytes = (size_t)DST_SP * DST_C * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(out_chw_host, d_chw_f32, out_bytes,
                    cudaMemcpyDeviceToHost, stream));

    /* Sync — caller reads out_chw_host immediately on return */
    cudaError_t syncErr = cudaStreamSynchronize(stream);
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaStreamSynchronize failed in %s: %s\n",
                __FUNCTION__, cudaGetErrorString(syncErr));
    }
}

} /* extern "C" */
