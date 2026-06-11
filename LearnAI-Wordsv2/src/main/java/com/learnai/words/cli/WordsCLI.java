package com.learnai.words.cli;

import com.learnai.words.nn.GpuLanguageModel;
import com.learnai.words.nn.TextGenerator;
import com.learnai.words.tokenizer.BPETokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.Collectors;

public class WordsCLI {
    private static final Logger logger = LoggerFactory.getLogger(WordsCLI.class);

    // Hyperparameters (overridable via -D JVM flags)
    private static final int   BLOCK_SIZE    = Integer.getInteger("block.size", 64);
    private static final int   D_MODEL       = Integer.getInteger("d.model", 128);
    private static final float LEARNING_RATE = Float.parseFloat(System.getProperty("learning.rate", "0.0005"));
    // MAX_EPOCHS is a safety cap only — training stops earlier via early stopping
    private static final int   MAX_EPOCHS    = Integer.getInteger("epochs", 200);

    // Validation: run every VAL_INTERVAL epochs; stop if val loss fails to improve EARLY_STOP_PATIENCE times in a row
    private static final int VAL_INTERVAL        = 5;
    private static final int EARLY_STOP_PATIENCE = 2;

    private interface ModelWrapper extends AutoCloseable {
        int  getCompletedEpochs();
        void setCompletedEpochs(int n);
        void close();
        float train(int[] input, int[] targets, float lr);
        float evaluate(int[] input, int[] targets);
        void save(String path) throws IOException;
        void load(String path) throws IOException;
        TextGenerator createGenerator(BPETokenizer tokenizer, int blockSize);
    }

    public static void main(String[] args) throws IOException {
        logger.info("--- Phase 2: Smart Student Training (BPE + 18 Layers) ---");

        // ── TOKENIZER ─────────────────────────────────────────────────────────
        BPETokenizer tokenizer = new BPETokenizer();
        Path tokPath = Path.of("tokenizer.bin");
        if (Files.exists(tokPath)) {
            logger.info("Loading BPE Tokenizer...");
            tokenizer.load(tokPath.toString());
        } else {
            logger.error("tokenizer.bin not found! Run BPETrainTool first.");
            return;
        }

        // ── CORPUS LOADING ────────────────────────────────────────────────────
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

        // ── MODEL SETUP ───────────────────────────────────────────────────────
        logger.info("Using GPU Model for training");
        GpuLanguageModel gpuModel = new GpuLanguageModel(tokenizer.getVocabSize(), D_MODEL, BLOCK_SIZE);
        final ModelWrapper model = new ModelWrapper() {
            public int  getCompletedEpochs()                         { return gpuModel.getCompletedEpochs(); }
            public void setCompletedEpochs(int n)                    { gpuModel.setCompletedEpochs(n); }
            public float train(int[] input, int[] targets, float lr)  { return gpuModel.train(input, targets, lr); }
            public float evaluate(int[] input, int[] targets)         { return gpuModel.evaluate(input, targets); }
            public void save(String path) throws IOException          { gpuModel.save(path); }
            public void load(String path) throws IOException          { gpuModel.load(path); }
            public TextGenerator createGenerator(BPETokenizer tok, int blockSize) {
                return new TextGenerator(gpuModel, tok, blockSize);
            }
            public void close() { gpuModel.close(); }
        };

        Path modelPath = Path.of("model.bin");
        int startEpoch = 1;
        if (Files.exists(modelPath)) {
            logger.info("Resuming from existing model...");
            model.load(modelPath.toString());
            startEpoch = model.getCompletedEpochs() + 1;
            logger.info("Resuming from Epoch {}", startEpoch);
        }

        // ── SEQUENCE PREPARATION ──────────────────────────────────────────────
        record SequencePair(int[] input, int target) {}
        List<SequencePair> allPairs = new ArrayList<>();
        for (int i = 0; i < allTokens.length - BLOCK_SIZE - 1; i += 10) {
            int[] seq = new int[BLOCK_SIZE];
            System.arraycopy(allTokens, i, seq, 0, BLOCK_SIZE);
            allPairs.add(new SequencePair(seq, allTokens[i + BLOCK_SIZE]));
        }

        // ── 90 / 10 TRAIN / VALIDATION SPLIT ─────────────────────────────────
        // Split is done BEFORE any shuffling so the validation set is always
        // drawn from the final 10% of the corpus — sequences the model never
        // trains on.
        int totalPairs = Math.min(allPairs.size(), 100000);
        int valSize    = Math.max(1, (int) (totalPairs * 0.10));
        int trainSize  = totalPairs - valSize;

        final List<SequencePair> trainPairs = new ArrayList<>(allPairs.subList(0, trainSize));
        final List<SequencePair> valPairs   = Collections.unmodifiableList(allPairs.subList(trainSize, totalPairs));

        logger.info("Train sequences: {}  |  Validation sequences: {}", trainSize, valSize);

        // ── EARLY STOPPING STATE ──────────────────────────────────────────────
        double bestValLoss = Double.MAX_VALUE;
        int    valWorseCount = 0;

        int trainBatchSize = Integer.getInteger("batch.size.train", 512);

        int epoch = startEpoch;
        try {
            while (true) {
                // Safety cap — prevents runaway training if early stopping never triggers
                if (MAX_EPOCHS > 0 && epoch > MAX_EPOCHS) {
                    logger.info("Reached safety cap of {} epochs. Stopping.", MAX_EPOCHS);
                    break;
                }

                // ── TRAINING ─────────────────────────────────────────────────
                final DoubleAdder   totalLoss = new DoubleAdder();
                final AtomicInteger processed = new AtomicInteger(0);
                final long          epochStart = System.currentTimeMillis();
                final int           currentEpoch = epoch;

                Thread heartbeat = new Thread(() -> {
                    try {
                        while (processed.get() < trainSize) {
                            Thread.sleep(10000);
                            int    count   = processed.get();
                            double elapsed = (System.currentTimeMillis() - epochStart) / 1000.0;
                            double tput    = (elapsed > 0) ? (count / elapsed) : 0;
                            logger.info("Heartbeat - Epoch {}: {}/{} (Throughput: {} seq/s)",
                                currentEpoch, count, trainSize, String.format("%.1f", tput));
                        }
                    } catch (InterruptedException ignored) {}
                });
                heartbeat.setDaemon(true);
                heartbeat.start();

                List<SequencePair> shuffledTrain = new ArrayList<>(trainPairs);
                Collections.shuffle(shuffledTrain);

                int numBatches = (trainSize + trainBatchSize - 1) / trainBatchSize;
                for (int b = 0; b < numBatches; b++) {
                    int startIdx         = b * trainBatchSize;
                    int endIdx           = Math.min(startIdx + trainBatchSize, trainSize);
                    int currentBatchSize = endIdx - startIdx;
                    if (currentBatchSize <= 0) break;

                    int[] flatInputs  = new int[currentBatchSize * BLOCK_SIZE];
                    int[] flatTargets = new int[currentBatchSize];
                    for (int i = 0; i < currentBatchSize; i++) {
                        SequencePair pair = shuffledTrain.get(startIdx + i);
                        System.arraycopy(pair.input(), 0, flatInputs, i * BLOCK_SIZE, BLOCK_SIZE);
                        flatTargets[i] = pair.target();
                    }

                    float loss = model.train(flatInputs, flatTargets, LEARNING_RATE);
                    totalLoss.add(loss * currentBatchSize);
                    processed.addAndGet(currentBatchSize);
                }

                long   epochEnd = System.currentTimeMillis();
                double avgTrainLoss = totalLoss.doubleValue() / trainSize;
                logger.info("Epoch {} Complete. Train Loss: {}. Time: {}s",
                    epoch, String.format("%.4f", avgTrainLoss), (epochEnd - epochStart) / 1000);

                model.setCompletedEpochs(epoch);

                // ── VALIDATION (every VAL_INTERVAL epochs) ─
                if (epoch % VAL_INTERVAL == 0) {

                    DoubleAdder valTotalLoss = new DoubleAdder();
                    int numValBatches = (valSize + trainBatchSize - 1) / trainBatchSize;

                    for (int b = 0; b < numValBatches; b++) {
                        int startIdx         = b * trainBatchSize;
                        int endIdx           = Math.min(startIdx + trainBatchSize, valSize);
                        int currentBatchSize = endIdx - startIdx;
                        if (currentBatchSize <= 0) break;

                        int[] flatInputs  = new int[currentBatchSize * BLOCK_SIZE];
                        int[] flatTargets = new int[currentBatchSize];
                        for (int i = 0; i < currentBatchSize; i++) {
                            SequencePair pair = valPairs.get(startIdx + i);
                            System.arraycopy(pair.input(), 0, flatInputs, i * BLOCK_SIZE, BLOCK_SIZE);
                            flatTargets[i] = pair.target();
                        }

                        // evaluate() is forward-only — no backward, no weight updates
                        float loss = model.evaluate(flatInputs, flatTargets);
                        valTotalLoss.add(loss * currentBatchSize);
                    }

                    double avgValLoss = valTotalLoss.doubleValue() / valSize;
                    logger.info("Epoch {} Validation Loss: {}  (Train Loss: {})",
                        epoch, String.format("%.4f", avgValLoss), String.format("%.4f", avgTrainLoss));

                    // Save checkpoint and generate text sample
                    model.save(modelPath.toString());
                    TextGenerator gen = model.createGenerator(tokenizer, BLOCK_SIZE);
                    logger.info("Sample (Epoch {}): [{}]", epoch, gen.generate("The ", 50));

                    // ── EARLY STOPPING CHECK ──────────────────────────────────
                    if (avgValLoss < bestValLoss) {
                        bestValLoss   = avgValLoss;
                        valWorseCount = 0;
                        logger.info("Epoch {} ✓ New best validation loss: {}", epoch, String.format("%.4f", bestValLoss));
                    } else {
                        valWorseCount++;
                        logger.warn("Epoch {} ⚠ Val loss did not improve ({}/{} patience). Best: {}  Current: {}",
                            epoch, valWorseCount, EARLY_STOP_PATIENCE,
                            String.format("%.4f", bestValLoss), String.format("%.4f", avgValLoss));

                        if (valWorseCount >= EARLY_STOP_PATIENCE) {
                            logger.warn("Early stopping triggered at Epoch {}. Best validation loss: {}",
                                epoch, String.format("%.4f", bestValLoss));
                            epoch = Integer.MAX_VALUE; // signal outer while to exit
                            break;
                        }
                    }
                }

                epoch++;
            }

            logger.info("Training complete. Final model saved to {}", modelPath);

        } catch (Exception e) {
            logger.error("Training interrupted", e);
        } finally {
            try {
                model.close();
            } catch (Exception e) {
                logger.error("Failed to close model", e);
            }
        }
    }
}
