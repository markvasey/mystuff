package org;

import org.junit.Test;

import org.junit.jupiter.api.Assertions;
import utils.Credential;

public class TestCredential {

    @Test
    public void TestCredentialConstructor() {
        Credential credential = new Credential("foo","bar");
        Assertions.assertEquals("foo", credential.getUsername());
        Assertions.assertEquals("bar", credential.getPassword());
    }
}
