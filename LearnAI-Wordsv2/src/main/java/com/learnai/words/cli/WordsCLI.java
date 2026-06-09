package com.learnai.words.cli;

import com.learnai.words.nn.LanguageModel;
import com.learnai.words.nn.GpuLanguageModel;
import com.learnai.words.nn.TextGenerator;
import com.learnai.words.tokenizer.BPETokenizer;
import com.learnai.words.tokenizer.TextDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;

public class WordsCLI {
    private static final Logger logger = LoggerFactory.getLogger(WordsCLI.class);
    
    // Phase 2 Hyperparameters
    private static final int BLOCK_SIZE = Integer.getInteger("block.size", 64);
    private static final int D_MODEL = Integer.getInteger("d.model", 128);
    private static final float LEARNING_RATE = Float.parseFloat(System.getProperty("learning.rate", "0.0005"));
    private static final int EPOCHS = Integer.getInteger("epochs", 40);
    private static final int THREADS = Integer.getInteger("threads", 14);
    private static final boolean USE_GPU = Boolean.getBoolean("use.gpu");

    private interface ModelWrapper extends AutoCloseable {
        int getCompletedEpochs();
        void setCompletedEpochs(int n);
        float train(int[] input, int target, float lr);
        void save(String path) throws IOException;
        void load(String path) throws IOException;
        TextGenerator createGenerator(BPETokenizer tokenizer, int blockSize);
        @Override
        void close();
    }

    public static void main(String[] args) throws IOException {
        logger.info("--- Phase 2: Smart Student Training (BPE + 18 Layers) ---");

        BPETokenizer tokenizer = new BPETokenizer();
        Path tokPath = Path.of("tokenizer.bin");
        if (Files.exists(tokPath)) {
            logger.info("Loading BPE Tokenizer...");
            tokenizer.load(tokPath.toString());
        } else {
            logger.error("tokenizer.bin not found! Run BPETrainTool first.");
            return;
        }

        String trainingDir = System.getProperty("training.dir", "Training/TinyStories");
        logger.info("Scanning directory: {}", trainingDir);
        List<Path> trainingFiles = Files.list(Path.of(trainingDir))
                .filter(p -> p.toString().endsWith(".txt"))
                .collect(Collectors.toList());

        StringBuilder corpusBuilder = new StringBuilder();
        for (Path f : trainingFiles) corpusBuilder.append(Files.readString(f)).append("\n");
        String fullText = corpusBuilder.toString();
        
        int[] allTokens = tokenizer.encode(fullText);
        logger.info("Total tokens in corpus: {}", allTokens.length);

        logger.info("Vocabulary Size: {}", tokenizer.getVocabSize());

        final ModelWrapper model;
        if (USE_GPU) {
            logger.info("Using GPU Model for training");
            GpuLanguageModel gpuModel = new GpuLanguageModel(tokenizer.getVocabSize(), D_MODEL, BLOCK_SIZE);
            model = new ModelWrapper() {
                public int getCompletedEpochs() { return gpuModel.getCompletedEpochs(); }
                public void setCompletedEpochs(int n) { gpuModel.setCompletedEpochs(n); }
                public float train(int[] input, int target, float lr) { return gpuModel.train(input, target, lr); }
                public void save(String path) throws IOException { gpuModel.save(path); }
                public void load(String path) throws IOException { gpuModel.load(path); }
                public TextGenerator createGenerator(BPETokenizer tokenizer, int blockSize) {
                    return new TextGenerator(gpuModel, tokenizer, blockSize);
                }
                public void close() { gpuModel.close(); }
            };
        } else {
            logger.info("Using CPU Model for training");
            LanguageModel cpuModel = new LanguageModel(tokenizer.getVocabSize(), D_MODEL, BLOCK_SIZE);
            model = new ModelWrapper() {
                public int getCompletedEpochs() { return cpuModel.getCompletedEpochs(); }
                public void setCompletedEpochs(int n) { cpuModel.setCompletedEpochs(n); }
                public float train(int[] input, int target, float lr) { return cpuModel.train(input, target, lr); }
                public void save(String path) throws IOException { cpuModel.save(path); }
                public void load(String path) throws IOException { cpuModel.load(path); }
                public TextGenerator createGenerator(BPETokenizer tokenizer, int blockSize) {
                    return new TextGenerator(cpuModel, tokenizer, blockSize);
                }
                public void close() {}
            };
        }

        Path modelPath = Path.of("model.bin");
        int startEpoch = 1;
        if (Files.exists(modelPath)) {
            logger.info("Resuming from existing model...");
            model.load(modelPath.toString());
            startEpoch = model.getCompletedEpochs() + 1;
            logger.info("Resuming from Epoch {}", startEpoch);
        }

        // Prepare training sequences efficiently
        record SequencePair(int[] input, int target) {}
        List<SequencePair> pairs = new ArrayList<>();
        for (int i = 0; i < allTokens.length - BLOCK_SIZE - 1; i += 10) {
            int[] seq = new int[BLOCK_SIZE];
            System.arraycopy(allTokens, i, seq, 0, BLOCK_SIZE);
            pairs.add(new SequencePair(seq, allTokens[i + BLOCK_SIZE]));
        }
        
        int totalSequences = Math.min(pairs.size(), 100000);
        final List<SequencePair> trainingBatch = pairs.subList(0, totalSequences);
        logger.info("Training on {} sequences", totalSequences);

        ForkJoinPool pool = new ForkJoinPool(THREADS);
        try {
            for (int epoch = startEpoch; epoch <= EPOCHS; epoch++) {
                final DoubleAdder totalLoss = new DoubleAdder();
                final AtomicInteger processed = new AtomicInteger(0);
                final long epochStart = System.currentTimeMillis();
                final int currentEpoch = epoch;
                
                // High-frequency Heartbeat Thread
                Thread heartbeat = new Thread(() -> {
                    try {
                        while (processed.get() < totalSequences) {
                            Thread.sleep(10000);
                            int count = processed.get();
                            double elapsed = (System.currentTimeMillis() - epochStart) / 1000.0;
                            double tput = (elapsed > 0) ? (count / elapsed) : 0;
                            logger.info("Heartbeat - Epoch {}: {}/{} (Throughput: {} seq/s)", 
                                currentEpoch, count, totalSequences, String.format("%.1f", tput));
                        }
                    } catch (InterruptedException ignored) {}
                });
                heartbeat.setDaemon(true);
                heartbeat.start();

                List<SequencePair> shuffledBatch = new ArrayList<>(trainingBatch);
                java.util.Collections.shuffle(shuffledBatch);

                if (USE_GPU) {
                    for (SequencePair pair : shuffledBatch) {
                        float loss = model.train(pair.input(), pair.target(), LEARNING_RATE);
                        totalLoss.add(loss);
                        
                        int count = processed.incrementAndGet();
                        if (count % 10000 == 0) {
                            try {
                                synchronized(model) { 
                                    model.setCompletedEpochs(currentEpoch - 1); // Partial save
                                    model.save(modelPath.toString()); 
                                }
                                logger.info("Checkpoint saved at {} sequences", count);
                            } catch (IOException e) { logger.error("Save failed", e); }
                        }
                    }
                } else {
                    pool.submit(() -> 
                        shuffledBatch.parallelStream().forEach(pair -> {
                            float loss = model.train(pair.input(), pair.target(), LEARNING_RATE);
                            totalLoss.add(loss);
                            
                            int count = processed.incrementAndGet();
                            if (count % 10000 == 0) {
                                try {
                                    synchronized(model) { 
                                        model.setCompletedEpochs(currentEpoch - 1); // Partial save
                                        model.save(modelPath.toString()); 
                                    }
                                    logger.info("Checkpoint saved at {} sequences", count);
                                } catch (IOException e) { logger.error("Save failed", e); }
                            }
                        })
                    ).get();
                }

                long epochEnd = System.currentTimeMillis();
                double avgLoss = totalLoss.doubleValue() / totalSequences;
                logger.info("Epoch {} Complete. Avg Loss: {}. Time: {}s", 
                    epoch, String.format("%.4f", avgLoss), (epochEnd - epochStart) / 1000);
                
                model.setCompletedEpochs(epoch);
                model.save(modelPath.toString());
                
                // Generate sample every epoch
                TextGenerator gen = model.createGenerator(tokenizer, BLOCK_SIZE);
                logger.info("Sample (Epoch {}): [{}]", epoch, gen.generate("The ", 50));
            }
        } catch (Exception e) {
            logger.error("Training interrupted", e);
        } finally {
            pool.shutdown();
            try {
                model.close();
            } catch (Exception e) {
                logger.error("Failed to close model", e);
            }
        }
    }
}
