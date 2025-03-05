package org.example.utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class AppProperties {

    public static String GetProperty(String propertyName) {
        String propertyValue = null;

        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        try (InputStream inputStream = classloader.getResourceAsStream("app.properties")) {
            Properties appProps = new Properties();
            try {
                if (inputStream != null) {
                    appProps.load(inputStream);
                    propertyValue = appProps.getProperty(propertyName);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        return propertyValue;
    }
}
