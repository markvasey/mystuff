package com.tapoviewer.cli;

import com.tapoviewer.math.CudaBridge;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class CudaLoadTester {
    public static void main(String[] args) {
        System.out.println("CudaLoadTester: Starting tight CUDA kernel loop on RTX 5060 Ti...");
        
        int width = 1920;
        int height = 1080;
        int size = width * height;
        
        // Allocate off-heap memory segments for test frames
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment prev = arena.allocate(ValueLayout.JAVA_BYTE, size);
            MemorySegment curr = arena.allocate(ValueLayout.JAVA_BYTE, size);
            
            // Fill arrays with test patterns
            for (int i = 0; i < size; i++) {
                prev.set(ValueLayout.JAVA_BYTE, i, (byte) (i % 256));
                curr.set(ValueLayout.JAVA_BYTE, i, (byte) ((i + 13) % 256));
            }
            
            System.out.println("Running 8,000 CUDA optical flow calculations on the GPU...");
            System.out.println("Check 'nvtop' or 'nvidia-smi' now!");
            
            long startTime = System.currentTimeMillis();
            int iterations = 8000;
            float dummySum = 0;
            
            for (int i = 0; i < iterations; i++) {
                float motion = CudaBridge.calculateMotionMagnitude(prev, curr, width, height);
                dummySum += motion;
                
                if (i % 1000 == 0 && i > 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    System.out.printf("[%d ms] Completed %d iterations. Current motion: %.4f\n", elapsed, i, motion);
                }
            }
            
            long totalTime = System.currentTimeMillis() - startTime;
            System.out.printf("Done! Completed %d iterations in %d ms (Avg: %.3f ms per frame).\n", 
                iterations, totalTime, (double) totalTime / iterations);
        } catch (Exception e) {
            System.err.println("CUDA execution failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
