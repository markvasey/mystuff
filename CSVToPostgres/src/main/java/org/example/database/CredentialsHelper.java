package org.example.database;

import org.example.utils.Files;

public class CredentialsHelper {
    public static Credentials LoadCredentials(String resourceFileName) {
        String fileContents = Files.ReadResourcesFile(resourceFileName);

        String[] fileContentsParts = fileContents.split(";");
        String[] usernameParts = fileContentsParts[0].split("=");
        String[] passwordParts = fileContentsParts[1].split("=");

        String username = usernameParts[1].replaceAll("(\\r|\\n)", "");
        String password = passwordParts[1].replaceAll("(\\r|\\n)", "");

        return new Credentials(username,password);
    }
}
