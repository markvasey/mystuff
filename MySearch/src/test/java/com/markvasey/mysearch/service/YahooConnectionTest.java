package com.markvasey.mysearch.service;

import jakarta.mail.*;
import org.junit.jupiter.api.Test;
import java.util.Properties;

public class YahooConnectionTest {

    @Test
    public void testYahooConnectionAndFolderList() throws Exception {
        // NOTE: These must match your application-local.properties
        String username = "markvasey@ymail.com";
        String appPassword = "ujlbawsfrjifzycv";
        String host = "imap.mail.yahoo.com";
        String port = "993";

        System.out.println("Connecting to Yahoo IMAP...");
        Properties props = new Properties();
        props.put("mail.store.protocol", "imaps");
        props.put("mail.imaps.host", host);
        props.put("mail.imaps.port", port);
        props.put("mail.imaps.ssl.enable", "true");
        props.put("mail.imaps.timeout", "10000"); // 10s timeout
        props.put("mail.imaps.connectiontimeout", "10000");

        Session session = Session.getInstance(props);
        Store store = session.getStore("imaps");
        
        try {
            store.connect(username, appPassword);
            System.out.println("Connected successfully!");

            Folder root = store.getDefaultFolder();
            listFolders(root, "");

        } finally {
            store.close();
        }
    }

    private void listFolders(Folder folder, String indent) throws MessagingException {
        Folder[] folders = folder.list();
        for (Folder f : folders) {
            System.out.println(indent + "Folder: " + f.getFullName() + " (Type: " + f.getType() + ")");
            if ((f.getType() & Folder.HOLDS_FOLDERS) != 0) {
                listFolders(f, indent + "  ");
            }
        }
    }
}
