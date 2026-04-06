package org.example.onepassword.utils;

import java.util.HashMap;

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
}
