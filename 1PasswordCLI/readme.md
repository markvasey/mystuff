# 1Password CLI Reader

A Spring Boot console application designed to locally read, search, and decrypt 1Password `.opvault` vaults. 
This tool allows you to access your credentials directly from the vault files without needing the official 1Password client or an internet connection.

## 1Password OpVault Design

The `OpVault` format is a directory-based storage system used by 1Password for local vaults. It is designed with "encryption at rest" in mind, 
ensuring that even if your vault files are stolen, your data remains secure without the master password.

### Structure
An `.opvault` directory (typically named `1Password.opvault`) contains:
- **`profile.js`**: Contains the vault's metadata, including the salt, iteration count for key derivation, and the encrypted Master Keys.
- **`band_X.js`**: (where X is 0-F) These files contain the actual vault items. Items are distributed across 16 "bands" based on the first character of their UUID.
- **`folders.js`**: Contains the folder structure of the vault.

### The `opdata01` Container
Most sensitive data in an OpVault is stored in a container format called `opdata01`. It consists of:
1.  **Magic Number (8 bytes)**: The literal string `opdata01`.
2.  **Plaintext Length (8 bytes)**: The length of the original data before encryption (Little Endian).
3.  **Initialization Vector (IV) (16 bytes)**: A random block used to initialize the AES encryption.
4.  **Ciphertext (Variable)**: The actual data encrypted with AES-256-CBC.
5.  **HMAC (32 bytes)**: A SHA-256 signature of the previous components to ensure the data hasn't been tampered with.

---

## Decryption Process

Decrypting a 1Password vault follows a hierarchical path, starting with your Master Password and ending with the plain JSON of your item details.

### 1. Deriving the Master Unlock Key
The program starts by taking your **Master Password** and combining it with a **Salt** (a random string found in `profile.js`). 
- **Method**: PBKDF2-HMAC-SHA512.
- **Iterations**: Usually 100,000 or more (defined in `profile.js`).
- **Result**: A 64-byte **Master Unlock Key**.

### 2. Unlocking the Vault Keys
The `profile.js` file contains two important encrypted blobs: `masterKey` and `overviewKey`.
- We use the **Master Unlock Key** to decrypt these blobs.
- The resulting raw keys are then hashed using **SHA-512** to produce the final 64-byte **Vault Master Key** and **Vault Overview Key**.
- These keys are split into two 32-byte halves: one for **AES-256-CBC Encryption** and one for **HMAC-SHA256 Integrity**.

### 3. Decrypting Item Overviews
Each item in a `band_X.js` file has an `o` field.
- This is an `opdata01` container encrypted with the **Vault Overview Key**.
- Once decrypted, it reveals a JSON object containing the **Title**, **URL**, and **Tags**. 
- This allows the program to search for items by name without decrypting the actual passwords.

### 4. Decrypting Item Keys
Unlike the overview, the actual **Item Details** (passwords, notes, etc.) are encrypted with a unique key for *every single item*.
- Each item has a `k` field. This is an encrypted blob containing the **Item Keys**.
- We use the **Vault Master Key** to decrypt the `k` field.
- The result is a unique 64-byte **Item Key** (32 bytes for encryption, 32 bytes for MAC) specific to that one item.

### 5. Decrypting Item Details
Finally, the `d` field contains the **Item Details**.
- This is an `opdata01` container encrypted with the **Item Key** derived in the previous step.
- Once decrypted, it reveals the full JSON structure including usernames, passwords, custom fields, and secure notes.

---

## Class Documentation

### `OnePasswordCliApplication`
The entry point of the application. It orchestrates the entire flow:
1.  Locates the vault path.
2.  Prompts the user for the Master Password securely.
3.  Calls the processors to parse the profile, derive keys, and search through the bands.

### `OpVaultDecryptor`
The cryptographic engine of the program. It implements:
- **PBKDF2 Key Derivation**: Specifically using HMAC-SHA512.
- **opdata01 Decryption**: Handles the verification of the HMAC signature and the AES-CBC decryption.
- **Item Decryption**: Logic to first decrypt the unique item keys and then use them to reveal the details.

### `VaultProfileProcessor`
Responsible for reading and parsing the `profile.js` file. It extracts the security parameters (salt, iterations) and 
the encrypted vault keys required to start the decryption process.

### `VaultKeysProcessor`
Manages the Master Keys. It takes the user's password and the profile data to derive and store the 
**Vault Master Key** and **Vault Overview Key** in memory while the program is running.

### `VaultBandsProcessor`
Handles the file system logic for iterating through the 16 band files (`band_0.js` through `band_F.js`). 
It reads each file, extracts the item collection, and passes them to the display logic.

### `DisplayProcessor`
The "UI" and filtering logic.
For every item found in the bands:
1.  It uses the **Overview** to get the title.
2.  It checks if the title matches the user's search terms.
3.  If it's a match, it proceeds to show the **Details** (passwords and fields).
4.  It outputs to the console, showing categories, titles, usernames, and passwords.

### `dataClasses` Package
Contains POJOs (Plain Old Java Objects) used by Jackson to map the JSON structures found in the vault files into Java objects 
(e.g., `VaultItem`, `ItemOverview`, `ItemDetails`, `ItemField`).

### `utils` Package
- **`ByteUtils`**: Helper for hex conversions.
- **`VaultPathUtils`**: Handles command-line arguments and file path resolution for the vault.
- **`Utils`**: Contains category mappings (e.g., mapping "001" to "Login") and search string parsing logic.

---

## References & Credits

- [1Password OpVault Documentation](https://support.1password.com/cs/opvault-design/) - Official design specification.
- [1Password Local Vaults (DarthNull)](https://darthnull.org/1pass-local-vaults/) - Excellent deep dive into the encryption details.
- [Go implementation (Evan T. Byrne)](https://github.com/evantbyrne/1password-opvault/blob/main/opvault/item.go) - Reference for logic validation.
- [Python implementation (Mickael Perrin)](https://github.com/mickaelperrin/onepassword-local-search) - Reference for logic validation.

![Key Derivation Flow](https://darthnull.org/media/2018/11/old_key_derivation.png) - Graphical representation of the OpVault keys.
