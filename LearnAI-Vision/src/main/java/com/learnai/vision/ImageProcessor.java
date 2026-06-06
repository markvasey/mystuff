package com.learnai.vision;

import com.learnai.math.Matrix;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Loads and processes images for the Vision Transformer.
 */
public class ImageProcessor {
    private final int imageSize; // e.g., 64 for 64x64 images
    private final int patchSize; // e.g., 8 for 8x8 patches

    public ImageProcessor(int imageSize, int patchSize) {
        this.imageSize = imageSize;
        this.patchSize = patchSize;
    }

    /**
     * Loads an image, resizes it, and converts it into a sequence of patches.
     * Each row in the returned Matrix is one flattened patch.
     */
    public Matrix imageToPatches(File imageFile) throws IOException {
        BufferedImage originalImage = ImageIO.read(imageFile);
        
        // Resize image to standard size
        BufferedImage resizedImage = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, imageSize, imageSize, null);
        g.dispose();

        int numPatchesSide = imageSize / patchSize;
        int numPatches = numPatchesSide * numPatchesSide;
        int patchDim = patchSize * patchSize * 3; // 3 for RGB channels

        Matrix patches = new Matrix(numPatches, patchDim);

        int patchIdx = 0;
        for (int y = 0; y < numPatchesSide; y++) {
            for (int x = 0; x < numPatchesSide; x++) {
                // Extract patch pixels
                double[] patchData = extractPatch(resizedImage, x * patchSize, y * patchSize);
                for (int i = 0; i < patchDim; i++) {
                    patches.set(patchIdx, i, patchData[i]);
                }
                patchIdx++;
            }
        }
        return patches;
    }

    private double[] extractPatch(BufferedImage img, int startX, int startY) {
        double[] data = new double[patchSize * patchSize * 3];
        int idx = 0;
        for (int y = 0; y < patchSize; y++) {
            for (int x = 0; x < patchSize; x++) {
                Color color = new Color(img.getRGB(startX + x, startY + y));
                // Normalize pixels to [0, 1] range
                data[idx++] = color.getRed() / 255.0;
                data[idx++] = color.getGreen() / 255.0;
                data[idx++] = color.getBlue() / 255.0;
            }
        }
        return data;
    }
}
