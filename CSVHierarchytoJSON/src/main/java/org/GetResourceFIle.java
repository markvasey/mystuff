package org;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class GetResourceFIle {
    public static List<String[]>  ReadCSV(String filename) {
        List<String[]> data = new ArrayList<>();

        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream inputStream = classloader.getResourceAsStream(filename);
        if(inputStream != null){
            InputStreamReader streamReader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);

            BufferedReader br = new BufferedReader(streamReader);
            String line;

            try {
                while ((line = br.readLine()) != null) {
                    String[] row = line.split(",");
                    data.add(row);
                }
                br.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        return data;
    }

}

