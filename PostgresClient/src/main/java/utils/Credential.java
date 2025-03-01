package utils;

import lombok.Getter;

/**
 *
 */
public class Credential {
    @Getter
    public String Username;
    @Getter
    public String Password;

    public Credential(String username, String password) {
        Username = username;
        Password = password;
    }
}
