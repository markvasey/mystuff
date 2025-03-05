package org.example.utils;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class Files {
    static public String ReadResourcesFile(String resourceFileName) {
        StringBuilder sb = new StringBuilder();
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream inputStream = classloader.getResourceAsStream(resourceFileName);
        if (inputStream != null) {
            InputStreamReader streamReader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);

            try (BufferedReader br = new BufferedReader(streamReader)) {
                String line = br.readLine();

                while (line != null) {
                    sb.append(line);
                    sb.append(System.lineSeparator());
                    line = br.readLine();
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        return sb.toString();
    }
}
