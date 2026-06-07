package com.learnai.words.tokenizer;
import java.io.IOException;

public class InspectTokenizer {
    public static void main(String[] args) throws IOException {
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.load("tokenizer.bin");
        System.out.println("Vocab Size: " + tokenizer.getVocabSize());
        
        // Let's sample some tokens from the end of the vocabulary (the most common merges)
        int vocabSize = tokenizer.getVocabSize();
        for (int i = vocabSize - 50; i < vocabSize; i++) {
            int[] id = {i};
            System.out.println("ID " + i + ": [" + tokenizer.decode(id) + "]");
        }
        
        // Also check some common English fragments
        String test = " the and of that which";
        int[] encoded = tokenizer.encode(test);
        System.out.print("Encoding '"+test+"': ");
        for(int e : encoded) System.out.print(e + " ");
        System.out.println("\nDecoded: [" + tokenizer.decode(encoded) + "]");
    }
}
