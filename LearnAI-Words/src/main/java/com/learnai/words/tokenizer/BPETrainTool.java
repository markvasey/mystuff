package com.learnai.words.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

public class BPETrainTool {
    public static void main(String[] args) throws IOException {
        System.out.println("--- Phase 2: BPE Vocabulary Discovery ---");
        
        List<Path> trainingFiles = Files.list(Path.of("Training"))
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
        // Training to 1,000 tokens (256 base + 744 merges)
        tokenizer.train(fullCorpus, 1000);

        tokenizer.save("tokenizer.bin");
        System.out.println("Successfully saved 1,000 tokens to tokenizer.bin");
    }
}
