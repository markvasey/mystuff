    ![Key Derivation](https://darthnull.org/media/2018/11/old_key_derivation.png)
    
Decrypting "k" data
 - the keys for the item details "d"

            [1Password OpVault documentation](https://support.1password.com/cs/opvault-design/)

            Data: 64 bytes
                typedef struct {
                  uint8_t crypto_key[32];
                  uint8_t mac_key[32];
                };
            IV: The data before the MAC is the AES-CBC encrypted item keys using unique random 16-byte IV. - 16 bytes
            MAC: The last 32 bytes comprise the HMAC-SHA256 of the IV and the encrypted data. The MAC is computed with the master MAC key. - 32 bytes

            [1Password Local Vaults](https://darthnull.org/1pass-local-vaults/)

            The key_data structure includes four components:

                Initialization Vector (IV) (16 bytes)
                Item Encryption Key (32 bytes)
                Item HMAC Key (32 bytes)
                HMAC Tag (32 bytes)
            First, compute the HMAC tag using the encrypted item keys (encryption + HMAC) as the message, and the Master HMAC Key as key.
            If that matches the HMAC tag found in the structure, then we know it hasn’t been altered.
            Now, use the Master AES key to decrypt the item keys, and those keys (AES + HMAC) to decrypt the actual vault item.

            See also: [Go implementation](https://github.com/evantbyrne/1password-opvault/blob/main/opvault/item.go)
                      [Python implementation](https://github.com/mickaelperrin/onepassword-local-search)
         