package com.learnai.words.cli;

import com.learnai.words.math.Matrix;
import com.learnai.words.nn.LanguageModel;
import com.learnai.words.nn.TextGenerator;
import com.learnai.words.tokenizer.CharacterTokenizer;
import com.learnai.words.tokenizer.TextDataset;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;

public class WordsCLI {
    private static final Logger logger = LoggerFactory.getLogger(WordsCLI.class);
    private static final int BLOCK_SIZE = 32;
    private static final int D_MODEL = 128;
    private static final double LEARNING_RATE = 0.001;
    private static final int EPOCHS = 50;
    private static final int THREADS = 14; // Multi-threaded production

    public static void main(String[] args) throws IOException {
        logger.info("--- LearnAI-Words LLM Training (Ultra-Responsive) ---");

        List<Path> trainingFiles = Files.list(Path.of("Training"))
                .filter(p -> p.toString().endsWith(".txt"))
                .collect(Collectors.toList());

        TextDataset dataset = new TextDataset(trainingFiles);
        CharacterTokenizer tokenizer = dataset.getTokenizer();
        
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), D_MODEL, BLOCK_SIZE);
        Path modelPath = Path.of("model.bin");
        if (Files.exists(modelPath)) {
            logger.info("Loading existing model...");
            model.load(modelPath.toString());
        }

        TextGenerator generator = new TextGenerator(model, tokenizer, BLOCK_SIZE);
        final List<TextDataset.SequencePair> allSequences = dataset.getSequences(BLOCK_SIZE);
        final List<TextDataset.SequencePair> sequences = allSequences.subList(0, Math.min(allSequences.size(), 50000));
        
        int totalSequences = sequences.size();
        logger.info("Total training sequences: {}", totalSequences);

        ForkJoinPool customThreadPool = new ForkJoinPool(THREADS);
        try {
            for (int epoch = 1; epoch <= EPOCHS; epoch++) {
                final DoubleAdder totalLoss = new DoubleAdder();
                final AtomicInteger processed = new AtomicInteger(0);
                long start = System.currentTimeMillis();
                
                logger.info("Starting Epoch {}", epoch);
                final int currentEpoch = epoch;
                final int tSeq = totalSequences;
                final long epochStart = System.currentTimeMillis();
                
                customThreadPool.submit(() -> 
                    sequences.parallelStream().forEach(pair -> {
                        long startStep = System.nanoTime();
                        double loss = model.train(pair.input(), pair.target(), LEARNING_RATE);
                        long endStep = System.nanoTime();
                        
                        totalLoss.add(loss);
                        
                        int count = processed.incrementAndGet();
                        if (count <= 100 && count % 10 == 0) {
                            logger.info("Avg Step Time (last 10): {}ms", (endStep - startStep) / 1_000_000);
                        }
                        if (count % 1000 == 0) {
                            long elapsed = System.currentTimeMillis() - epochStart;
                            logger.info("Epoch {}: Processed {}/{} ({}s elapsed)", currentEpoch, count, tSeq, elapsed / 1000);
                        }
                        if (count % 10000 == 0) {
                            try {
                                synchronized(model) {
                                    model.save(modelPath.toString());
                                }
                                logger.info("Checkpoint saved at {} sequences", count);
                            } catch (IOException e) {
                                logger.error("Failed to save checkpoint", e);
                            }
                        }
                    })
                ).get();

                long end = System.currentTimeMillis();
                double avgLoss = totalLoss.doubleValue() / totalSequences;
                logger.info("Epoch {} Complete - Avg Loss: {} - Time: {}s", 
                    epoch, String.format("%.4f", avgLoss), (end - start) / 1000);
                
                logger.info("Sample: [{}]", generator.generate("The ", 100));
                model.save(modelPath.toString());
            }
        } catch (Exception e) {
            logger.error("Training interrupted", e);
            Thread.currentThread().interrupt();
        } finally {
            customThreadPool.shutdown();
        }
    }
}
