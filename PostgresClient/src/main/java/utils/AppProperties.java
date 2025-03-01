package utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Properties;

public class AppProperties {

    public static String GetProperty(String propertyName) {
        URL url = Thread.currentThread().getContextClassLoader().getResource("app.properties");

        Properties appProps = new Properties();
        try {
            if (url != null) {
                appProps.load(new FileInputStream(url.getPath()));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return appProps.getProperty(propertyName);
    }
}
