package org.example.database;

import lombok.Getter;

@Getter
public class Credentials {
    private final String Username;
    private final String Password;


    public Credentials(String username, String password) {
        Username = username;
        Password = password;
    }
}
