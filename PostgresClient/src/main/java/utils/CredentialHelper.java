package utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Properties;

public class CredentialHelper {
    public static Credential LoadCredential() {
        URL url = Thread.currentThread().getContextClassLoader().getResource("creds.txt");

        Properties appProps = new Properties();
        try {
            if (url != null) {
                appProps.load(new FileInputStream(url.getPath()));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        String username = appProps.getProperty("username");
        String password = appProps.getProperty("password");

        return new Credential(username,password);
    }
}
