package com.learnai.words.nn;

import com.learnai.words.tokenizer.CharacterTokenizer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class GenerationTest {
    private LanguageModel model;
    private CharacterTokenizer tokenizer;
    private TextGenerator generator;
    private static final int BLOCK_SIZE = 32;

    @BeforeEach
    public void setup() {
        // Use a small corpus to initialize the tokenizer
        String corpus = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"-";
        tokenizer = new CharacterTokenizer(corpus);
        model = new LanguageModel(tokenizer.getVocabSize(), 64, BLOCK_SIZE);
        generator = new TextGenerator(model, tokenizer, BLOCK_SIZE);
    }

    @Test
    public void testDorianGrayCompletion() {
        String prompt = "The artist is ";
        String result = generator.generate(prompt, 20);
        
        assertTrue(result.startsWith(prompt), "Result should start with the prompt");
        assertTrue(result.length() > prompt.length(), "Result should be longer than the prompt");
        System.out.println("Dorian Gray Sample: " + result);
    }

    @Test
    public void testGreatGatsbyCompletion() {
        String prompt = "In my younger and ";
        String result = generator.generate(prompt, 20);
        
        assertTrue(result.startsWith(prompt));
        assertTrue(result.length() > prompt.length());
        System.out.println("Great Gatsby Sample: " + result);
    }

    @Test
    public void testFrankensteinCompletion() {
        String prompt = "I am by birth a ";
        String result = generator.generate(prompt, 20);
        
        assertTrue(result.startsWith(prompt));
        assertTrue(result.length() > prompt.length());
        System.out.println("Frankenstein Sample: " + result);
    }

    @Test
    public void testSherlockHolmesCompletion() {
        String prompt = "To Sherlock Holmes she ";
        String result = generator.generate(prompt, 20);
        
        assertTrue(result.startsWith(prompt));
        assertTrue(result.length() > prompt.length());
        System.out.println("Sherlock Holmes Sample: " + result);
    }
}
