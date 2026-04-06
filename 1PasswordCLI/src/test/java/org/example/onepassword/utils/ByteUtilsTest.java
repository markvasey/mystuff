package org.example.onepassword.utils;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ByteUtilsTest {

    @Test
    void testBytesToHexEmpty() {
        assertEquals("", ByteUtils.bytesToHex(new byte[0]));
    }

    @Test
    void testBytesToHexSingleByte() {
        assertEquals("00", ByteUtils.bytesToHex(new byte[]{0x00}));
        assertEquals("0F", ByteUtils.bytesToHex(new byte[]{0x0F}));
        assertEquals("10", ByteUtils.bytesToHex(new byte[]{0x10}));
        assertEquals("FF", ByteUtils.bytesToHex(new byte[]{(byte) 0xFF}));
    }

    @Test
    void testBytesToHexMultipleBytes() {
        byte[] input = new byte[]{0x01, 0x23, 0x45, 0x67, (byte) 0x89, (byte) 0xAB, (byte) 0xCD, (byte) 0xEF};
        assertEquals("0123456789ABCDEF", ByteUtils.bytesToHex(input));
    }

    @Test
    void testBytesToHexNull() {
        assertThrows(NullPointerException.class, () -> ByteUtils.bytesToHex(null));
    }
}
