package com.learnai.words.tokenizer;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CharacterTokenizerTest {

    @Test
    public void testEncodeDecode() {
        String corpus = "abcdefg";
        CharacterTokenizer tokenizer = new CharacterTokenizer(corpus);
        
        assertEquals(7, tokenizer.getVocabSize());
        
        String input = "face";
        int[] encoded = tokenizer.encodeString(input);
        String decoded = tokenizer.decodeIds(encoded);
        
        assertEquals(input, decoded);
    }

    @Test
    public void testUnknownChar() {
        CharacterTokenizer tokenizer = new CharacterTokenizer("abc");
        assertEquals('?', tokenizer.decode(999));
    }
}
