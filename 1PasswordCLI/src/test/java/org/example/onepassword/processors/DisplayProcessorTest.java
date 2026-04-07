package org.example.onepassword.processors;

import org.example.onepassword.dataClasses.VaultItem;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.Mac;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DisplayProcessorTest {

    private byte[] vaultMasterKey;
    private byte[] vaultOverviewKey;
    private Map<String, VaultItem> testItems;

    @BeforeEach
    void setUp() throws Exception {
        DisplayProcessor.resetCounters();
        vaultMasterKey = new byte[64];
        Arrays.fill(vaultMasterKey, (byte) 0x11);
        vaultOverviewKey = new byte[64];
        Arrays.fill(vaultOverviewKey, (byte) 0x22);

        testItems = new HashMap<>();
        // 1. Login item (2 fields)
        testItems.put("UUID1", createMockItem("001", "Google Login", "https://google.com", 
            "{\"fields\":[{\"name\":\"username\",\"value\":\"user1\"},{\"name\":\"password\",\"value\":\"pass1\"}]}", 
            vaultMasterKey, vaultOverviewKey));

        // 2. Credit Card (1 field + number)
        testItems.put("UUID2", createMockItem("002", "My Visa Card", null, 
            "{\"fields\":[{\"name\":\"cardholder\",\"value\":\"Mark\"}], \"number\":\"1234-5678\"}", 
            vaultMasterKey, vaultOverviewKey));

        // 3. Secure Note (0 fields, has notesPlain)
        testItems.put("UUID3", createMockItem("003", "Private Note", null, 
            "{\"notesPlain\":\"Secret stuff\"}", 
            vaultMasterKey, vaultOverviewKey));
    }

    @Test
    void testDisplayResultsAll() {
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, null);
        assertEquals(3, DisplayProcessor.getCountTitles());
        assertEquals(3, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsBlankSearch() {
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "  ");
        assertEquals(3, DisplayProcessor.getCountTitles());
        assertEquals(3, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsSearchGoogle() {
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "Google");
        assertEquals(1, DisplayProcessor.getCountTitles());
        assertEquals(2, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsSearchCaseInsensitive() {
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "visa");
        assertEquals(1, DisplayProcessor.getCountTitles());
        assertEquals(1, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsSearchNoMatch() {
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "Amazon");
        assertEquals(0, DisplayProcessor.getCountTitles());
        assertEquals(0, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsSearchMultipleWords() {
        // "Google Card" should match UUID1 (Google) and UUID2 (Visa Card)
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "Google Card");
        assertEquals(2, DisplayProcessor.getCountTitles());
        assertEquals(3, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsSearchQuotedPhrase() {
        // "\"Google Login\"" should only match UUID1
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "\"Google Login\"");
        assertEquals(1, DisplayProcessor.getCountTitles());
        assertEquals(2, DisplayProcessor.getCountFields());
    }

    @Test
    void testDisplayResultsSearchQuotedAndUnquoted() {
        // "\"Private Note\" Google" should match UUID3 and UUID1
        DisplayProcessor.DisplayResults(testItems, vaultMasterKey, vaultOverviewKey, "\"Private Note\" Google");
        assertEquals(2, DisplayProcessor.getCountTitles());
        assertEquals(2, DisplayProcessor.getCountFields());
    }

    private VaultItem createMockItem(String category, String title, String url, String detailsJson, byte[] vMasterKey, byte[] vOverviewKey) throws Exception {
        String overviewJson = "{\"title\":\"" + title + "\"" + (url != null ? ",\"url\":\"" + url + "\"" : "") + "}";
        String encryptedO = Base64.getEncoder().encodeToString(encryptOpData(overviewJson, vOverviewKey));

        byte[] itemKeys = new byte[64];
        Arrays.fill(itemKeys, (byte) 0x33);
        String encryptedK = Base64.getEncoder().encodeToString(encryptItemKey(itemKeys, vMasterKey));

        String encryptedD = Base64.getEncoder().encodeToString(encryptOpData(detailsJson, itemKeys));

        VaultItem item = new VaultItem();
        item.setCategory(category);
        item.setO(encryptedO);
        item.setK(encryptedK);
        item.setD(encryptedD);
        item.setUuid("MOCK-" + title);
        return item;
    }

    private byte[] encryptOpData(String json, byte[] keys) throws Exception {
        byte[] plainBytes = json.getBytes(StandardCharsets.UTF_8);
        byte[] encKey = Arrays.copyOfRange(keys, 0, 32);
        byte[] macKey = Arrays.copyOfRange(keys, 32, 64);
        byte[] iv = new byte[16];
        Arrays.fill(iv, (byte) 0x01);

        int paddedLen = ((plainBytes.length + 15) / 16) * 16;
        byte[] paddedPlaintext = new byte[paddedLen];
        System.arraycopy(plainBytes, 0, paddedPlaintext, paddedLen - plainBytes.length, plainBytes.length);

        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(encKey, "AES"), new IvParameterSpec(iv));
        byte[] ciphertext = cipher.doFinal(paddedPlaintext);

        int totalLen = 8 + 8 + 16 + ciphertext.length + 32;
        ByteBuffer buffer = ByteBuffer.allocate(totalLen).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("opdata01".getBytes(StandardCharsets.US_ASCII));
        buffer.putLong(plainBytes.length);
        buffer.put(iv);
        buffer.put(ciphertext);

        Mac hmac = Mac.getInstance("HmacSHA256");
        hmac.init(new SecretKeySpec(macKey, "HmacSHA256"));
        hmac.update(buffer.array(), 0, totalLen - 32);
        buffer.put(hmac.doFinal());

        return buffer.array();
    }

    private byte[] encryptItemKey(byte[] itemKeys, byte[] vMasterKey) throws Exception {
        byte[] vEncKey = Arrays.copyOfRange(vMasterKey, 0, 32);
        byte[] vMacKey = Arrays.copyOfRange(vMasterKey, 32, 64);
        byte[] iv = new byte[16];
        Arrays.fill(iv, (byte) 0x02);

        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(vEncKey, "AES"), new IvParameterSpec(iv));
        byte[] ciphertext = cipher.doFinal(itemKeys);

        int totalLen = 16 + ciphertext.length + 32;
        ByteBuffer buffer = ByteBuffer.allocate(totalLen);
        buffer.put(iv);
        buffer.put(ciphertext);

        Mac hmac = Mac.getInstance("HmacSHA256");
        hmac.init(new SecretKeySpec(vMacKey, "HmacSHA256"));
        hmac.update(buffer.array(), 0, totalLen - 32);
        buffer.put(hmac.doFinal());

        return buffer.array();
    }
}
