package com.learnai.words.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

public class BPETrainTool {
    public static void main(String[] args) throws IOException {
        System.out.println("--- Phase 2: BPE Vocabulary Discovery ---");
        
        String trainingDir = System.getProperty("training.dir", "Training/TinyStories");
        int vocabSize = Integer.getInteger("vocab.size", 800);
        
        System.out.println("Scanning directory: " + trainingDir);
        List<Path> trainingFiles = Files.list(Path.of(trainingDir))
                .filter(p -> p.toString().endsWith(".txt"))
                .collect(Collectors.toList());

        StringBuilder corpusBuilder = new StringBuilder();
        for (Path file : trainingFiles) {
            String content = Files.readString(file);
            corpusBuilder.append(TextDataset.extractBookText(content)).append("\n\n");
        }
        
        String fullCorpus = corpusBuilder.toString();
        System.out.println("Corpus loaded: " + fullCorpus.length() + " characters.");

        BPETokenizer tokenizer = new BPETokenizer();
        // Training to target vocab tokens
        tokenizer.train(fullCorpus, vocabSize);

        tokenizer.save("tokenizer.bin");
        System.out.println("Successfully saved " + vocabSize + " tokens to tokenizer.bin");
    }
}
