package com.learnai.words.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TextDataset {
    private final String fullText;
    private final CharacterTokenizer tokenizer;

    public TextDataset(List<Path> files) throws IOException {
        StringBuilder sb = new StringBuilder();
        for (Path file : files) {
            String content = Files.readString(file);
            sb.append(extractBookText(content)).append("\n\n");
        }
        this.fullText = sb.toString();
        this.tokenizer = new CharacterTokenizer(fullText);
    }

    public TextDataset(String text) {
        this.fullText = text;
        this.tokenizer = new CharacterTokenizer(fullText);
    }

    public static String extractBookText(String content) {
        // Look for the Gutenberg markers
        Pattern startPattern = Pattern.compile("\\*\\*\\* START OF THE PROJECT GUTENBERG EBOOK .* \\*\\*\\*");
        Pattern endPattern = Pattern.compile("\\*\\*\\* END OF THE PROJECT GUTENBERG EBOOK .* \\*\\*\\*");

        Matcher startMatcher = startPattern.matcher(content);
        Matcher endMatcher = endPattern.matcher(content);

        int start = 0;
        if (startMatcher.find()) {
            start = startMatcher.end();
        }

        int end = content.length();
        if (endMatcher.find()) {
            end = endMatcher.start();
        }

        if (start < end) {
            return content.substring(start, end).trim();
        }
        return content.trim();
    }

    public String getFullText() {
        return fullText;
    }

    public CharacterTokenizer getTokenizer() {
        return tokenizer;
    }

    public List<SequencePair> getSequences(int blockSize) {
        int[] ids = tokenizer.encodeString(fullText);
        List<SequencePair> pairs = new ArrayList<>();
        for (int i = 0; i < ids.length - blockSize; i++) {
            int[] input = new int[blockSize];
            System.arraycopy(ids, i, input, 0, blockSize);
            int target = ids[i + blockSize];
            pairs.add(new SequencePair(input, target));
        }
        return pairs;
    }

    public static record SequencePair(int[] input, int target) {}
}
