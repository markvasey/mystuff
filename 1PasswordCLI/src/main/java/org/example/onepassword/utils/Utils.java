package org.example.onepassword.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Utils {

    private static HashMap<String, String> categories = new HashMap<>();

    static {
        categories.put("001", "Login");
        categories.put("002", "Credit Card");
        categories.put("003", "Secure Note");
        categories.put("004", "Identity");
        categories.put("005", "Password");
        categories.put("099", "Tombstone");
        categories.put("100", "Software License");
        categories.put("101", "Bank Account");
        categories.put("102", "Database");
        categories.put("103", "Driver License");
        categories.put("104", "Outdoor License");
        categories.put("105", "Membership");
        categories.put("106", "Passport");
        categories.put("107", "Rewards");
        categories.put("108", "SSN");
        categories.put("109", "Router");
        categories.put("110", "Server");
        categories.put("111", "Email");
        categories.put("112", "Document");
    }

    public static String getCategoryName(String categoryCode) {
        if (categoryCode == null) {
            return null;
        }
        return categories.getOrDefault(categoryCode, categoryCode);
    }

    public static List<String> parseSearchString(String searchString) {
        List<String> terms = new ArrayList<>();
        if (searchString == null || searchString.isBlank()) {
            return terms;
        }

        // Regex to match "quoted phrases" or individual words
        Pattern pattern = Pattern.compile("\"([^\"]*)\"|(\\S+)");
        Matcher matcher = pattern.matcher(searchString);

        while (matcher.find()) {
            if (matcher.group(1) != null) {
                // Quoted phrase
                terms.add(matcher.group(1));
            } else {
                // Individual word
                terms.add(matcher.group(2));
            }
        }
        return terms;
    }
}
