package com.learnai.words.tokenizer;

import java.io.*;
import java.util.*;

/**
 * Byte Pair Encoding (BPE) Tokenizer.
 * Discovers and merges the most frequent character pairs to form subword "word parts".
 */
public class BPETokenizer {
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final List<int[]> merges = new ArrayList<>();
    private int vocabSize;
    private static final int UNK_ID = 256;

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
        List<Integer> ids = new ArrayList<>();
        for (char c : corpus.toCharArray()) {
            int cid = (int) c;
            ids.add(cid < 256 ? cid : UNK_ID);
        }

        while (vocabSize < targetVocabSize) {
            Map<String, Integer> stats = getStats(ids);
            if (stats.isEmpty()) break;
            
            String bestPairStr = Collections.max(stats.entrySet(), Map.Entry.comparingByValue()).getKey();
            String[] parts = bestPairStr.split(",");
            int[] bestPair = {Integer.parseInt(parts[0]), Integer.parseInt(parts[1])};

            int newTokenId = vocabSize++;
            merges.add(bestPair);
            
            String t1 = idToToken.get(bestPair[0]);
            String t2 = idToToken.get(bestPair[1]);
            String newTokenStr = (t1 == null ? "" : t1) + (t2 == null ? "" : t2);
            
            idToToken.put(newTokenId, newTokenStr);
            tokenToId.put(newTokenStr, newTokenId);

            ids = merge(ids, bestPair, newTokenId);
            if (vocabSize % 100 == 0) System.out.println("Vocab Size: " + vocabSize);
        }
    }

    private Map<String, Integer> getStats(List<Integer> ids) {
        Map<String, Integer> stats = new HashMap<>();
        for (int i = 0; i < ids.size() - 1; i++) {
            String pair = ids.get(i) + "," + ids.get(i + 1);
            stats.put(pair, stats.getOrDefault(pair, 0) + 1);
        }
        return stats;
    }

    private List<Integer> merge(List<Integer> ids, int[] pair, int newTokenId) {
        List<Integer> newIds = new ArrayList<>(ids.size());
        for (int i = 0; i < ids.size(); i++) {
            if (i < ids.size() - 1 && ids.get(i) == pair[0] && ids.get(i + 1) == pair[1]) {
                newIds.add(newTokenId);
                i++;
            } else {
                newIds.add(ids.get(i));
            }
        }
        return newIds;
    }

    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        for (char c : text.toCharArray()) {
            int cid = (int) c;
            ids.add(cid < 256 ? cid : UNK_ID);
        }

        for (int[] pair : merges) {
            String s1 = idToToken.get(pair[0]);
            String s2 = idToToken.get(pair[1]);
            Integer mid = tokenToId.get(s1 + s2);
            if (mid != null) {
                ids = merge(ids, pair, mid);
            }
        }

        return ids.stream().mapToInt(i -> i).toArray();
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
            for (int i = 0; i < 256; i++) {
                String s = "" + (char) i;
                idToToken.put(i, s);
                tokenToId.put(s, i);
            }
            idToToken.put(UNK_ID, "<UNK>");
            tokenToId.put("<UNK>", UNK_ID);
            
            for (int i = 0; i < numMerges; i++) {
                int[] pair = {dis.readInt(), dis.readInt()};
                int newTokenId = 257 + i; // Offset by 256 + 1 (UNK)
                merges.add(pair);
                String t1 = idToToken.get(pair[0]);
                String t2 = idToToken.get(pair[1]);
                String newTokenStr = (t1 == null ? "" : t1) + (t2 == null ? "" : t2);
                idToToken.put(newTokenId, newTokenStr);
                tokenToId.put(newTokenStr, newTokenId);
            }
            this.vocabSize = targetVocabSize;
        }
    }
}
