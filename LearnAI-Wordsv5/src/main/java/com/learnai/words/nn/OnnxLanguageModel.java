package com.learnai.words.nn;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import java.nio.LongBuffer;
import java.util.Collections;

public class OnnxLanguageModel implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxLanguageModel(String modelPath) throws Exception {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    /**
     * Executes the forward pass of the ONNX model.
     * 
     * @param tokenIds sequence of input token IDs
     * @return float array representing the logits of the last token in the sequence
     */
    public float[] predict(int[] tokenIds) throws Exception {
        long[] shape = new long[]{1, tokenIds.length};
        long[] longTokenIds = new long[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            longTokenIds[i] = tokenIds[i];
        }

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longTokenIds), shape)) {
            // Run prediction on "input_ids" and obtain "logits" output
            try (OrtSession.Result result = session.run(Collections.singletonMap("input_ids", inputTensor))) {
                OnnxTensor outputTensor = (OnnxTensor) result.get(0);
                
                // Retrieve 3D value array: [batch_size][sequence_length][vocab_size]
                float[][][] logits = (float[][][]) outputTensor.getValue();
                
                // Extract only the logits for the very last token
                int lastTokenIdx = tokenIds.length - 1;
                return logits[0][lastTokenIdx];
            }
        }
    }

    @Override
    public void close() throws Exception {
        if (session != null) {
            session.close();
        }
        if (env != null) {
            env.close();
        }
    }
}
