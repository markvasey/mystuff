package com.learnai.words.tokenizer;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class BPETest {
    @Test
    public void testBPEBasic() {
        BPETokenizer tokenizer = new BPETokenizer();
        String corpus = "low lower newest widest";
        tokenizer.train(corpus, 300); // Small target for test

        String input = "lower";
        int[] encoded = tokenizer.encode(input);
        String decoded = tokenizer.decode(encoded);

        assertEquals(input, decoded);
        // "lower" should be fewer than 5 tokens if merges happened
        assertTrue(encoded.length < 5);
        System.out.println("Encoded '" + input + "' to " + encoded.length + " tokens");
    }

    @Test
    public void testBPESaveLoad(@org.junit.jupiter.api.io.TempDir java.nio.file.Path tempDir) throws java.io.IOException {
        BPETokenizer tokenizer1 = new BPETokenizer();
        String corpus = "low lower newest widest";
        tokenizer1.train(corpus, 280);

        String path = tempDir.resolve("tokenizer.bin").toString();
        tokenizer1.save(path);

        BPETokenizer tokenizer2 = new BPETokenizer();
        tokenizer2.load(path);

        assertEquals(tokenizer1.getVocabSize(), tokenizer2.getVocabSize());
        
        String testText = "newest widest";
        int[] enc1 = tokenizer1.encode(testText);
        int[] enc2 = tokenizer2.encode(testText);
        assertArrayEquals(enc1, enc2);
        assertEquals(tokenizer1.decode(enc1), tokenizer2.decode(enc2));
    }
}
