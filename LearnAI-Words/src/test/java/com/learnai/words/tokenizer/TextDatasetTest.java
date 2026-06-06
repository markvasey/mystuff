package com.learnai.words.tokenizer;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class TextDatasetTest {

    @Test
    public void testExtractBookText() {
        String content = "Preamble\n" +
                "*** START OF THE PROJECT GUTENBERG EBOOK THE PICTURE OF DORIAN GRAY ***\n" +
                "This is the actual story.\n" +
                "*** END OF THE PROJECT GUTENBERG EBOOK THE PICTURE OF DORIAN GRAY ***\n" +
                "Postamble";
        
        String extracted = TextDataset.extractBookText(content);
        assertEquals("This is the actual story.", extracted);
    }

    @Test
    public void testExtractBookTextNoMarkers() {
        String content = "Just some text without markers.";
        String extracted = TextDataset.extractBookText(content);
        assertEquals(content, extracted);
    }

    @Test
    public void testGetSequences() {
        String text = "1234567890"; // 10 chars
        TextDataset dataset = new TextDataset(text);
        int blockSize = 3;
        
        List<TextDataset.SequencePair> sequences = dataset.getSequences(blockSize);
        
        // With length 10 and block size 3, we should get 10 - 3 = 7 sequences
        // 123 -> 4, 234 -> 5, 345 -> 6, 456 -> 7, 567 -> 8, 678 -> 9, 789 -> 0
        assertEquals(7, sequences.size());
        
        // Verify first sequence
        assertArrayEquals(dataset.getTokenizer().encodeString("123"), sequences.get(0).input());
        assertEquals(dataset.getTokenizer().encode('4'), sequences.get(0).target());
    }
}
