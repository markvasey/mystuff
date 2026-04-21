package com.example.pmqsmonitor.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class APIGid {

    private static final Pattern GID_PATTERN = Pattern.compile("^(\\d{4}-\\d{2}-\\d{2})([a-z])\\.(\\d+)\\.(\\d+)$");

    public record GIdParts(String date, String letter, String number1, String number2) {
        @Override
        public String toString() {
            return String.format("%s%s.%s.%s", date, letter, number1, number2);
        }
    }

    /**
     * Parses a speech ID in the format YYYY-MM-DD[letter].[number1].[number2]
     * e.g., 2026-01-21c.298.4
     *
     * @param speechId The speech ID to parse
     * @return SpeechIdParts containing the separated components, or null if format doesn't match
     */
    public static GIdParts parseGId(String speechId) {
        if (speechId == null) {
            return null;
        }

        Matcher matcher = GID_PATTERN.matcher(speechId);
        if (matcher.matches()) {
            return new GIdParts(
                    matcher.group(1),
                    matcher.group(2),
                    matcher.group(3),
                    matcher.group(4)
            );
        }

        return null;
    }

    /**
     * Increments number2 in SpeechIdParts by 1.
     *
     * @param parts The SpeechIdParts to increment
     * @return A new SpeechIdParts with incremented number2
     */
    public static GIdParts incrementNumber2(GIdParts parts) {
        if (parts == null) {
            return null;
        }
        try {
            int n2 = Integer.parseInt(parts.number2());
            return new GIdParts(parts.date(), parts.letter(), parts.number1(), String.valueOf(n2 + 1));
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
