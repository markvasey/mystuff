package com.learnai.words.tokenizer;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Optimized Byte-Level Byte Pair Encoding (BBPE) Tokenizer.
 * Supports raw bytes (0-255) as the base vocabulary, completely eliminating <UNK> tokens.
 */
public class BPETokenizer {
    private final Map<Integer, byte[]> idToTokenBytes = new ConcurrentHashMap<>();
    private final List<int[]> merges = new ArrayList<>();
    private final Map<String, int[]> encodeCache = new ConcurrentHashMap<>();
    private int vocabSize;
    private static final Pattern WORD_PATTERN = Pattern.compile(" ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]|\\s+");

    public BPETokenizer() {
        // Initialize with basic raw bytes (0-255)
        for (int i = 0; i < 256; i++) {
            byte[] b = new byte[]{(byte) i};
            idToTokenBytes.put(i, b);
        }
        this.vocabSize = 256;
    }

    public void train(String corpus, int targetVocabSize) {
        System.out.println("Training BPE Tokenizer (Target Vocab: " + targetVocabSize + ")...");
        encodeCache.clear();

        // 1. Pre-tokenize the corpus into words and count their frequencies
        Matcher matcher = WORD_PATTERN.matcher(corpus);
        Map<String, Integer> wordFreqs = new HashMap<>();
        while (matcher.find()) {
            String word = matcher.group();
            wordFreqs.put(word, wordFreqs.getOrDefault(word, 0) + 1);
        }

        // 2. Represent each unique word as an array of byte IDs
        List<int[]> wordTokens = new ArrayList<>(wordFreqs.size());
        int[] wordCounts = new int[wordFreqs.size()];
        int idx = 0;
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            byte[] wordBytes = word.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            int[] ids = new int[wordBytes.length];
            for (int i = 0; i < wordBytes.length; i++) {
                ids[i] = wordBytes[i] & 0xFF;
            }
            wordTokens.add(ids);
            wordCounts[idx++] = freq;
        }

        // 3. Iteratively find and merge the most frequent pairs
        while (vocabSize < targetVocabSize) {
            Map<Long, Integer> stats = new HashMap<>();
            for (int j = 0; j < wordTokens.size(); j++) {
                int[] ids = wordTokens.get(j);
                int count = wordCounts[j];
                for (int i = 0; i < ids.length - 1; i++) {
                    long pair = ((long) ids[i] << 32) | (ids[i + 1] & 0xFFFFFFFFL);
                    stats.put(pair, stats.getOrDefault(pair, 0) + count);
                }
            }

            if (stats.isEmpty()) break;

            // Find the most frequent pair
            long bestPairKey = -1;
            int maxFreq = -1;
            for (Map.Entry<Long, Integer> entry : stats.entrySet()) {
                if (entry.getValue() > maxFreq) {
                    maxFreq = entry.getValue();
                    bestPairKey = entry.getKey();
                }
            }

            if (bestPairKey == -1) break;

            int left = (int) (bestPairKey >>> 32);
            int right = (int) bestPairKey;
            int newTokenId = vocabSize++;
            int[] bestPair = {left, right, newTokenId};

            merges.add(bestPair);

            byte[] t1 = idToTokenBytes.get(left);
            byte[] t2 = idToTokenBytes.get(right);
            byte[] newTokenBytes = concat(t1, t2);
            idToTokenBytes.put(newTokenId, newTokenBytes);

            // Apply merge to all unique words
            for (int j = 0; j < wordTokens.size(); j++) {
                int[] ids = wordTokens.get(j);
                wordTokens.set(j, mergeWord(ids, bestPair));
            }

            if (vocabSize % 100 == 0) {
                System.out.println("Vocab Size: " + vocabSize);
            }
        }
    }

    private byte[] concat(byte[] a, byte[] b) {
        if (a == null) return b == null ? new byte[0] : b;
        if (b == null) return a;
        byte[] res = new byte[a.length + b.length];
        System.arraycopy(a, 0, res, 0, a.length);
        System.arraycopy(b, 0, res, a.length, b.length);
        return res;
    }

    private int[] mergeWord(int[] ids, int[] pair) {
        int len = ids.length;
        int count = 0;
        for (int i = 0; i < len - 1; i++) {
            if (ids[i] == pair[0] && ids[i + 1] == pair[1]) {
                count++;
                i++;
            }
        }
        if (count == 0) return ids;

        int[] newIds = new int[len - count];
        int w = 0;
        for (int i = 0; i < len; i++) {
            if (i < len - 1 && ids[i] == pair[0] && ids[i + 1] == pair[1]) {
                newIds[w++] = pair[2];
                i++;
            } else {
                newIds[w++] = ids[i];
            }
        }
        return newIds;
    }

    public int[] encode(String text) {
        Matcher matcher = WORD_PATTERN.matcher(text);
        List<String> words = new ArrayList<>();
        while (matcher.find()) {
            words.add(matcher.group());
        }

        // Encode unique words in parallel stream, flattening into single int array
        return words.parallelStream()
                .map(this::encodeWord)
                .flatMapToInt(Arrays::stream)
                .toArray();
    }

    private int[] encodeWord(String word) {
        int[] cached = encodeCache.get(word);
        if (cached != null) return cached;

        byte[] wordBytes = word.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int[] ids = new int[wordBytes.length];
        for (int i = 0; i < wordBytes.length; i++) {
            ids[i] = wordBytes[i] & 0xFF;
        }

        for (int[] pair : merges) {
            ids = mergeWord(ids, pair);
        }

        encodeCache.put(word, ids);
        return ids;
    }

    public String decode(int[] ids) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        for (int id : ids) {
            byte[] tokenBytes = idToTokenBytes.get(id);
            if (tokenBytes != null) {
                baos.write(tokenBytes, 0, tokenBytes.length);
            }
        }
        return baos.toString(java.nio.charset.StandardCharsets.UTF_8);
    }

    public int getVocabSize() { return vocabSize; }

    public void save(String path) throws IOException {
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(path))) {
            dos.writeInt(vocabSize);
            dos.writeInt(merges.size());
            for (int[] pair : merges) {
                dos.writeInt(pair[0]);
                dos.writeInt(pair[1]);
            }
        }
    }

    public void load(String path) throws IOException {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
            int targetVocabSize = dis.readInt();
            int numMerges = dis.readInt();

            // Clear current state and reset to base 256 bytes
            idToTokenBytes.clear();
            merges.clear();
            encodeCache.clear();
            
            for (int i = 0; i < 256; i++) {
                byte[] b = new byte[]{(byte) i};
                idToTokenBytes.put(i, b);
            }

            for (int i = 0; i < numMerges; i++) {
                int left = dis.readInt();
                int right = dis.readInt();
                int newTokenId = 256 + i; // Offset by 256 base bytes
                int[] pair = {left, right, newTokenId};
                merges.add(pair);
                
                byte[] t1 = idToTokenBytes.get(left);
                byte[] t2 = idToTokenBytes.get(right);
                byte[] newTokenBytes = concat(t1, t2);
                idToTokenBytes.put(newTokenId, newTokenBytes);
            }
            this.vocabSize = targetVocabSize;
        }
    }
}
