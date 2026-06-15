package com.learnai.words.tokenizer;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Optimized Byte Pair Encoding (BPE) Tokenizer.
 * Uses a pre-tokenized dictionary-based approach for training (thousands of times faster)
 * and parallel stream with a concurrent cache for encoding (sub-millisecond encoding).
 */
public class BPETokenizer {
    private final Map<Integer, String> idToToken = new ConcurrentHashMap<>();
    private final Map<String, Integer> tokenToId = new ConcurrentHashMap<>();
    private final List<int[]> merges = new ArrayList<>();
    private final Map<String, int[]> encodeCache = new ConcurrentHashMap<>();
    private int vocabSize;
    private static final int UNK_ID = 256;
    private static final Pattern WORD_PATTERN = Pattern.compile(" ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]|\\s+");

    public BPETokenizer() {
        // Initialize with basic ASCII/extended characters (0-255)
        for (int i = 0; i < 256; i++) {
            String s = "" + (char) i;
            idToToken.put(i, s);
            tokenToId.put(s, i);
        }
        // Add UNK token
        idToToken.put(UNK_ID, "<UNK>");
        tokenToId.put("<UNK>", UNK_ID);
        this.vocabSize = 257;
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

        // 2. Represent each unique word as an array of character IDs
        List<int[]> wordTokens = new ArrayList<>(wordFreqs.size());
        int[] wordCounts = new int[wordFreqs.size()];
        int idx = 0;
        for (Map.Entry<String, Integer> entry : wordFreqs.entrySet()) {
            String word = entry.getKey();
            int freq = entry.getValue();
            int[] ids = new int[word.length()];
            for (int i = 0; i < word.length(); i++) {
                int cid = (int) word.charAt(i);
                ids[i] = cid < 256 ? cid : UNK_ID;
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

            String t1 = idToToken.get(left);
            String t2 = idToToken.get(right);
            String newTokenStr = (t1 == null ? "" : t1) + (t2 == null ? "" : t2);

            idToToken.put(newTokenId, newTokenStr);
            tokenToId.put(newTokenStr, newTokenId);

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

        int[] ids = new int[word.length()];
        for (int i = 0; i < word.length(); i++) {
            int cid = (int) word.charAt(i);
            ids[i] = cid < 256 ? cid : UNK_ID;
        }

        for (int[] pair : merges) {
            ids = mergeWord(ids, pair);
        }

        encodeCache.put(word, ids);
        return ids;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(idToToken.getOrDefault(id, ""));
        }
        return sb.toString();
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

            // Clear current state and reset to base + UNK
            idToToken.clear();
            tokenToId.clear();
            merges.clear();
            encodeCache.clear();
            
            for (int i = 0; i < 256; i++) {
                String s = "" + (char) i;
                idToToken.put(i, s);
                tokenToId.put(s, i);
            }
            idToToken.put(UNK_ID, "<UNK>");
            tokenToId.put("<UNK>", UNK_ID);

            for (int i = 0; i < numMerges; i++) {
                int left = dis.readInt();
                int right = dis.readInt();
                int newTokenId = 257 + i; // Offset by 256 + 1 (UNK)
                int[] pair = {left, right, newTokenId};
                merges.add(pair);
                
                String t1 = idToToken.get(left);
                String t2 = idToToken.get(right);
                String newTokenStr = (t1 == null ? "" : t1) + (t2 == null ? "" : t2);
                idToToken.put(newTokenId, newTokenStr);
                tokenToId.put(newTokenStr, newTokenId);
            }
            this.vocabSize = targetVocabSize;
        }
    }
}
