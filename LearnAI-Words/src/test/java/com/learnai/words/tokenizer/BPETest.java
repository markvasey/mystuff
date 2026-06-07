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
}
