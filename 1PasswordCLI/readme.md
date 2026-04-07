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

## Example Walkthrough

To see how the math actually works, let's follow a single item from the test data through the decryption process.

### 1. The Inputs (from `profile.js`)
*   **Master Password**: `test-password`
*   **Salt**: `YmFzZTY0LXNhbHQ=` (Base64 for "base64-salt")
*   **Iterations**: `1000`

### 2. Deriving the Master Unlock Key
The program calls `OpVaultDecryptor.deriveKeys()`. Using PBKDF2-HMAC-SHA512, it calculates a 64-byte key. 
- **First 32 bytes**: Used for AES decryption of the Master Keys.
- **Last 32 bytes**: Used for HMAC integrity verification of the Master Keys.

### 3. Unlocking the Vault Master Key
In `profile.js`, the `masterKey` field might look like a Base64 string.
1.  We verify the HMAC of the `masterKey` blob using the last 32 bytes of our **Master Unlock Key**.
2.  We decrypt the ciphertext using the first 32 bytes of our **Master Unlock Key**.
3.  We hash the resulting 64 bytes using **SHA-512** to get our final **Vault Master Key**.

### 4. Decrypting an Item (from `band_X.js`)
Imagine we find an item with these fields:
*   **`k` (Encrypted Item Keys)**: A blob containing a unique key for this item.
*   **`d` (Encrypted Details)**: An `opdata01` container with the password data.

#### Step A: Decrypt the `k` field
1.  We take our **Vault Master Key**.
2.  We split it into its two 32-byte halves (**Encryption** and **MAC**).
3.  We verify the HMAC of the `k` blob and then decrypt the ciphertext.
4.  This reveals the **Item Key** (a 64-byte key unique to this specific item).

#### Step B: Decrypt the `d` field
1.  We take the **Item Key** from Step A and split it into its two 32-byte halves.
2.  We verify the HMAC signature at the end of the `d` blob using the **MAC half**.
3.  We use the IV (bytes 16-31 of the blob) and the **Encryption half** to decrypt the ciphertext.
4.  **Result**: The JSON string:
    `{"fields":[{"name":"username","value":"markvasey"},{"name":"password","value":"correct-horse-battery-staple"}]}`


### 5. Final Display
The `DisplayProcessor` parses this JSON and prints:
```text
Title: My Bank Login
  username = markvasey
  Password: correct-horse-battery-staple
```

---

## Introduction to Cryptography

To understand how 1Password secures your data, it's helpful to understand the basic building blocks of modern cryptography 
used in this application.

### AES (Advanced Encryption Standard)
AES is the industry-standard algorithm for encrypting data. It is a **Symmetric** cipher, meaning the same key is used to 
both encrypt and decrypt the data.
- **AES-256**: This application uses the strongest version (256-bit keys).
- **CBC (Cipher Block Chaining)**: A "mode" of operation where each block of plaintext is combined with the previous ciphertext block 
- before being encrypted. This ensures that identical blocks of text don't result in identical ciphertext.

### SHA (Secure Hash Algorithm)
A "Hash" is a one-way digital fingerprint. You can turn any data into a hash, but you cannot turn a hash back into the original data.
- **SHA-256 / SHA-512**: These produce 32-byte and 64-byte fingerprints, respectively.
- **Usage**: Used to verify data integrity and to "stretch" short passwords into long, complex cryptographic keys.

### HMAC (Hash-based Message Authentication Code)
An HMAC is like a signature for a piece of data. It uses a secret key and a hash function (like SHA-256) to prove that a file or message 
has not been altered by anyone who doesn't know the key. If even a single bit of the data is changed, the HMAC will no longer match.

### Salt
A **Salt** is a random string of data added to your password before it is hashed.
- **Purpose**: Without a salt, two people with the same password would have the same "Master Key." 
- This would make it easier for hackers to use "Rainbow Tables" (pre-computed lists of common password hashes) to steal data. 
- A unique salt ensures that every vault is cryptographically unique, even if the passwords are the same.

### IV (Initialization Vector)
An **IV** is a random block of data used to start the AES-CBC encryption process.
- **Purpose**: It ensures that if you encrypt the same piece of data twice with the same key, the resulting ciphertext will be completely 
- different each time. The IV does not need to be secret (it is stored in plain sight in the `opdata01` container), but it must be unique 
- for every encryption operation.

### Ciphertext vs. Plaintext
- **Plaintext**: The original, readable data (like your actual password or a JSON string).
- **Ciphertext**: The "scrambled" version of the data produced after encryption. Without the correct key and IV, ciphertext looks like 
- random noise.

### Key Derivation & Iterations
Since humans use short, memorable passwords and computers need long, random keys, we use **PBKDF2** (Password-Based Key Derivation Function 2).
- **Iterations**: This is the number of times the hashing process is repeated (e.g., 100,000 times). 
- By forcing the computer to do thousands of calculations, we make it much slower and more expensive for an attacker to 
- try and "guess" your password.

---

## References & Credits

- [1Password OpVault Documentation](https://support.1password.com/cs/opvault-design/) - Official design specification.
- [1Password Local Vaults (DarthNull)](https://darthnull.org/1pass-local-vaults/) - Excellent deep dive into the encryption details.
- [Go implementation (Evan T. Byrne)](https://github.com/evantbyrne/1password-opvault/blob/main/opvault/item.go) - Reference for logic validation.
- [Python implementation (Mickael Perrin)](https://github.com/mickaelperrin/onepassword-local-search) - Reference for logic validation.

![Key Derivation Flow](https://darthnull.org/media/2018/11/old_key_derivation.png) - Graphical representation of the OpVault keys.
