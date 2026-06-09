package com.learnai.words.tokenizer;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

public class CharacterTokenizer {
    private final Map<Character, Integer> charToId = new HashMap<>();
    private final Map<Integer, Character> idToChar = new HashMap<>();
    private final int vocabSize;

    public CharacterTokenizer(String text) {
        Set<Character> chars = new TreeSet<>();
        for (char c : text.toCharArray()) {
            chars.add(c);
        }
        int id = 0;
        for (char c : chars) {
            charToId.put(c, id);
            idToChar.put(id, c);
            id++;
        }
        this.vocabSize = id;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public int encode(char c) {
        return charToId.getOrDefault(c, 0); // Default to first char if unknown
    }

    public char decode(int id) {
        return idToChar.getOrDefault(id, '?');
    }

    public int[] encodeString(String s) {
        int[] ids = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            ids[i] = encode(s.charAt(i));
        }
        return ids;
    }

    public String decodeIds(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(decode(id));
        }
        return sb.toString();
    }
}
