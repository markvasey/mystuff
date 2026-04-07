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
        categories.put("Identity", "004");
        categories.put("Password", "005");
        categories.put("Tombstone", "099");
        categories.put("Software License", "100");
        categories.put("Bank Account", "101");
        categories.put("Database", "102");
        categories.put("Driver License", "103");
        categories.put("Outdoor License", "104");
        categories.put("Membership", "105");
        categories.put("Passport", "106");
        categories.put("Rewards", "107");
        categories.put("SSN", "108");
        categories.put("Router", "109");
        categories.put("Server", "110");
        categories.put("Email", "111");
    }

    public static String getCategoryName(String categoryCode) {
        return categories.get(categoryCode);
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
