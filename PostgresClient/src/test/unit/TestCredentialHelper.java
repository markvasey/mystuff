package unit;

import org.junit.Test;
import org.junit.jupiter.api.Assertions;
import utils.Credential;
import utils.CredentialHelper;

public class TestCredentialHelper {
    @Test
    public void TestCredentialConstructor() {
        Credential credential = CredentialHelper.LoadCredential();
        Assertions.assertNotNull(credential);
        Assertions.assertNotNull( credential.getUsername());
        Assertions.assertNotNull( credential.getPassword());
        Assertions.assertFalse( credential.getUsername().isEmpty());
        Assertions.assertFalse( credential.getPassword().isEmpty());
    }
}
