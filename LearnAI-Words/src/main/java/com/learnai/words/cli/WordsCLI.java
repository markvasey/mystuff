package com.learnai.words.cli;

import com.learnai.words.nn.LanguageModel;
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
    private static final int BLOCK_SIZE = 128;
    private static final int D_MODEL = 192;
    private static final double LEARNING_RATE = 0.0005;
    private static final int EPOCHS = 150;
    private static final int THREADS = 14;

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

        List<Path> trainingFiles = Files.list(Path.of("Training"))
                .filter(p -> p.toString().endsWith(".txt"))
                .collect(Collectors.toList());

        StringBuilder corpusBuilder = new StringBuilder();
        for (Path f : trainingFiles) corpusBuilder.append(Files.readString(f)).append("\n");
        String fullText = corpusBuilder.toString();
        
        int[] allTokens = tokenizer.encode(fullText);
        logger.info("Total tokens in corpus: {}", allTokens.length);

        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), D_MODEL, BLOCK_SIZE);
        logger.info("Vocabulary Size: {}", tokenizer.getVocabSize());
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

                pool.submit(() -> 
                    trainingBatch.parallelStream().forEach(pair -> {
                        double loss = model.train(pair.input(), pair.target(), LEARNING_RATE);
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

                long epochEnd = System.currentTimeMillis();
                double avgLoss = totalLoss.doubleValue() / totalSequences;
                logger.info("Epoch {} Complete. Avg Loss: {}. Time: {}s", 
                    epoch, String.format("%.4f", avgLoss), (epochEnd - epochStart) / 1000);
                
                model.setCompletedEpochs(epoch);
                model.save(modelPath.toString());
                
                // Generate sample every epoch
                TextGenerator gen = new TextGenerator(model, tokenizer, BLOCK_SIZE);
                logger.info("Sample (Epoch {}): [{}]", epoch, gen.generate("The ", 50));
            }
        } catch (Exception e) {
            logger.error("Training interrupted", e);
        } finally {
            pool.shutdown();
        }
    }
}
