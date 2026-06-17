package com.tapoviewer.math;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

public class CudaBridge {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;

    private static final MethodHandle calculateMotionMagnitudeHandle;
    private static final MethodHandle preprocessFrameForYoloHandle;   // Priority 3

    static {
        // Find the shared library libseizure_cuda.so
        Path libPath = Path.of("libseizure_cuda.so");
        if (!Files.exists(libPath)) {
            libPath = Path.of(System.getProperty("user.dir"), "libseizure_cuda.so");
        }
        if (!Files.exists(libPath)) {
            libPath = Path.of(System.getProperty("user.dir"), "src", "main", "native", "libseizure_cuda.so");
        }

        if (Files.exists(libPath)) {
            LOOKUP = SymbolLookup.libraryLookup(libPath.toAbsolutePath(), Arena.global());
        } else {
            throw new UnsatisfiedLinkError(
                "Could not find libseizure_cuda.so in working directory or target paths. " +
                "Please run 'make -C src/main/native' to compile it.");
        }

        try {
            // Priority 1+2: async-stream + NPP motion magnitude
            calculateMotionMagnitudeHandle = LINKER.downcallHandle(
                LOOKUP.find("calculate_motion_magnitude").orElseThrow(
                    () -> new NoSuchMethodError("calculate_motion_magnitude")),
                FunctionDescriptor.of(ValueLayout.JAVA_FLOAT,
                    ValueLayout.ADDRESS, // prev_host
                    ValueLayout.ADDRESS, // curr_host
                    ValueLayout.JAVA_INT,  // width
                    ValueLayout.JAVA_INT   // height
                )
            );

            // Priority 3: NPP YOLO frame preprocessing
            preprocessFrameForYoloHandle = LINKER.downcallHandle(
                LOOKUP.find("preprocess_frame_for_yolo").orElseThrow(
                    () -> new NoSuchMethodError("preprocess_frame_for_yolo")),
                FunctionDescriptor.ofVoid(
                    ValueLayout.ADDRESS,  // bgr_host  (source frame, HWC uint8)
                    ValueLayout.JAVA_INT, // src_w
                    ValueLayout.JAVA_INT, // src_h
                    ValueLayout.JAVA_INT, // src_step
                    ValueLayout.ADDRESS   // out_chw_host (3*640*640 floats, caller-owned)
                )
            );
        } catch (Exception e) {
            throw new RuntimeException("Failed to bind native CUDA functions", e);
        }
    }

    /** Priority 1+2: Motion magnitude using dedicated stream + NPP norm-diff. */
    public static float calculateMotionMagnitude(MemorySegment prevSegment,
                                                  MemorySegment currSegment,
                                                  int width, int height) {
        try {
            return (float) calculateMotionMagnitudeHandle.invokeExact(
                prevSegment, currSegment, width, height);
        } catch (Throwable ex) {
            throw new RuntimeException(
                "Error executing calculate_motion_magnitude: " + ex.getMessage(), ex);
        }
    }

    /**
     * Priority 3: GPU-accelerated YOLO input preprocessing.
     * <p>
     * Converts a {@code src_w × src_h} BGR uint8 frame (HWC layout) into a
     * {@code 640 × 640} RGB float32 CHW tensor normalised to [0, 1].
     * The result is written directly into {@code outChwHost}, which must be a
     * {@link java.nio.ByteBuffer#allocateDirect direct} ByteBuffer of at least
     * {@code 3 * 640 * 640 * 4} bytes.
     *
     * @param bgrHost    MemorySegment over the source BGR frame (pageable host memory)
     * @param srcW       source frame width  in pixels
     * @param srcH       source frame height in pixels
     * @param outChwHost MemorySegment over the caller's pre-allocated output buffer
     */
    public static void preprocessFrameForYolo(MemorySegment bgrHost,
                                               int srcW, int srcH,
                                               int srcStep,
                                               MemorySegment outChwHost) {
        try {
            preprocessFrameForYoloHandle.invokeExact(bgrHost, srcW, srcH, srcStep, outChwHost);
        } catch (Throwable ex) {
            throw new RuntimeException(
                "Error executing preprocess_frame_for_yolo: " + ex.getMessage(), ex);
        }
    }
}
