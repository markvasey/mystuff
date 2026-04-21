package com.example.pmqsmonitor.util;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class APIGidTest {

    @Test
    void testParseGId_ValidFormat() {
        // Arrange
        String speechId = "2026-01-21c.298.4";

        // Act
        APIGid.GIdParts parts = APIGid.parseGId(speechId);

        // Assert
        assertNotNull(parts);
        assertEquals("2026-01-21", parts.date());
        assertEquals("c", parts.letter());
        assertEquals("298", parts.number1());
        assertEquals("4", parts.number2());
    }

    @Test
    void testParseGId_InvalidFormat() {
        assertNull(APIGid.parseGId("invalid-format"));
        assertNull(APIGid.parseGId(null));
        assertNull(APIGid.parseGId("2026-01-21.298.4")); // Missing letter
    }

    @Test
    void testIncrementNumber2() {
        // Arrange
        APIGid.GIdParts parts = new APIGid.GIdParts("2026-01-21", "c", "298", "4");

        // Act
        APIGid.GIdParts incremented = APIGid.incrementNumber2(parts);

        // Assert
        assertNotNull(incremented);
        assertEquals("2026-01-21", incremented.date());
        assertEquals("c", incremented.letter());
        assertEquals("298", incremented.number1());
        assertEquals("5", incremented.number2());
    }

    @Test
    void testToString() {
        // Arrange
        APIGid.GIdParts parts = new APIGid.GIdParts("2026-01-21", "c", "298", "4");

        // Act & Assert
        assertEquals("2026-01-21c.298.4", parts.toString());
    }
}
