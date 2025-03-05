package org.example.utils;

import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class NumberUtils {
    private static final DateFormat DateFormat1 = new SimpleDateFormat("dd/MM/yyyy");

    public static boolean isInteger(String value) {
        try {
            value = value.replaceAll(",","");
            Integer.parseInt(value);
            return true;
        } catch (NumberFormatException e) {
            try {
                double d = Double.parseDouble(value);
                int i = (int) d;
                if (i == d) {
                    return true;
                }
            } catch (NumberFormatException exc) {
                return false;
            }
            return false;
        }
    }

    public static int getInteger(String value) {
        int returnValue = Integer.MIN_VALUE;
        try {
            value = value.replaceAll(",","");
            returnValue = Integer.parseInt(value);
        } catch (NumberFormatException e) {
            try {
                double d = Double.parseDouble(value);
                int i = (int) d;
                if (i == d) {
                    returnValue = i;
                }
            } catch (NumberFormatException exc) {
                //Ignore
            }
        }
        return returnValue;
    }

    public static boolean isBoolean(String value) {
        value = value.toLowerCase();
        return value.equals("yes") || value.equals("no") || value.equals("true") || value.equals("false") || value.equals("0") || value.equals("1")
                || Boolean.parseBoolean(value);
    }


    public static boolean getBoolean(String value) {
        value = value.toLowerCase();
        if( value.equals("yes") || value.equals("true") || value.equals("1")) {
            return true;
        } else {
            return Boolean.parseBoolean(value);
        }
    }
    public static boolean isDouble(String value) {
        try {
            value = value.replaceAll(",","").replaceAll("(\\d+)(\\.00)","");
            Double.parseDouble(value);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    public static boolean isDate(String value) {
        try {
            DateFormat1.parse(value);
            return true;
        } catch (NumberFormatException | ParseException e) {
            return false;
        }
    }

    public static Date getDate(String value) {
        try {
            return DateFormat1.parse(value);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    public static String removeTrailingZeroes(String s) {
        StringBuilder sb = new StringBuilder(s);
        while (!sb.isEmpty() && sb.charAt(sb.length() - 1) == '0') {
            sb.setLength(sb.length() - 1);
        }
        return sb.toString();
    }

    public static String removeTrailingZeroDecimalPlaces(String s) {
        StringBuilder sb = new StringBuilder(s);
        if(s.contains(".")) {
            while (!sb.isEmpty() && sb.charAt(sb.length() - 1) == '0') {
                sb.setLength(sb.length() - 1);
            }
            if(sb.charAt(sb.length() - 1) == '.') {
                sb.setLength(sb.length() - 1);
            }
        }
        return sb.toString();
    }
}
