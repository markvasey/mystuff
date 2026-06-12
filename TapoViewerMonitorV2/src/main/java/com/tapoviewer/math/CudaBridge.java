package com.tapoviewer.math;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

public class CudaBridge {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;

    private static final MethodHandle calculateMotionMagnitudeHandle;

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
            throw new UnsatisfiedLinkError("Could not find libseizure_cuda.so in working directory or target paths. " +
                    "Please run 'make -C src/main/native' to compile it.");
        }

        try {
            calculateMotionMagnitudeHandle = LINKER.downcallHandle(
                LOOKUP.find("calculate_motion_magnitude").orElseThrow(() -> new NoSuchMethodError("calculate_motion_magnitude")),
                FunctionDescriptor.of(ValueLayout.JAVA_FLOAT, 
                    ValueLayout.ADDRESS, // prev_host
                    ValueLayout.ADDRESS, // curr_host
                    ValueLayout.JAVA_INT,  // width
                    ValueLayout.JAVA_INT   // height
                )
            );
        } catch (Exception e) {
            throw new RuntimeException("Failed to bind native CUDA functions", e);
        }
    }

    public static float calculateMotionMagnitude(MemorySegment prevSegment, MemorySegment currSegment, int width, int height) {
        try {
            return (float) calculateMotionMagnitudeHandle.invokeExact(prevSegment, currSegment, width, height);
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing calculate_motion_magnitude: " + ex.getMessage(), ex);
        }
    }
}
