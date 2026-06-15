import struct
import regex as re

class BPETokenizer:
    UNK_ID = 256
    WORD_PATTERN = re.compile(r" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]|\s+")

    def __init__(self):
        self.id_to_token = {i: bytes([i]) for i in range(256)}
        self.id_to_token[self.UNK_ID] = b"<UNK>"
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.merges = []  # list of tuples (left, right)
        self.vocab_size = 257
        self.encode_cache = {}

    def train(self, corpus: str, target_vocab_size: int):
        print(f"Training BPE Tokenizer (Target Vocab: {target_vocab_size})...")
        self.encode_cache.clear()

        # 1. Pre-tokenize the corpus into words and count frequencies
        word_freqs = {}
        for match in self.WORD_PATTERN.finditer(corpus):
            word = match.group()
            word_freqs[word] = word_freqs.get(word, 0) + 1

        # 2. Represent each unique word as a list of character IDs
        word_tokens = []
        word_counts = []
        for word, freq in word_freqs.items():
            ids = []
            for char in word:
                cid = ord(char)
                ids.append(cid if cid < 256 else self.UNK_ID)
            word_tokens.append(ids)
            word_counts.append(freq)

        # 3. Iteratively find and merge the most frequent pairs
        while self.vocab_size < target_vocab_size:
            stats = {}
            for ids, count in zip(word_tokens, word_counts):
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    stats[pair] = stats.get(pair, 0) + count

            if not stats:
                break

            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break

            left, right = best_pair
            new_token_id = self.vocab_size
            self.vocab_size += 1

            self.merges.append(best_pair)

            t1 = self.id_to_token.get(left, b'')
            t2 = self.id_to_token.get(right, b'')
            # Merge strings
            new_token_str = t1 + t2
            self.id_to_token[new_token_id] = new_token_str
            self.token_to_id[new_token_str] = new_token_id

            # Apply merge to all unique words
            for j in range(len(word_tokens)):
                word_tokens[j] = self._merge_word(word_tokens[j], best_pair, new_token_id)

            if self.vocab_size % 100 == 0:
                print(f"Vocab Size: {self.vocab_size}")

    def _merge_word(self, ids, pair, new_id):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def encode(self, text: str):
        words = [m.group() for m in self.WORD_PATTERN.finditer(text)]
        result = []
        for w in words:
            result.extend(self._encode_word(w))
        return result

    def _encode_word(self, word: str):
        if word in self.encode_cache:
            return self.encode_cache[word]

        ids = []
        for char in word:
            cid = ord(char)
            ids.append(cid if cid < 256 else self.UNK_ID)

        for i, pair in enumerate(self.merges):
            new_id = 257 + i
            ids = self._merge_word(ids, pair, new_id)

        self.encode_cache[word] = ids
        return ids

    def decode(self, ids):
        parts = []
        for idx in ids:
            token_bytes = self.id_to_token.get(idx, b'')
            # Decode bytes back to string, replacing errors to prevent crashing
            parts.append(token_bytes.decode('utf-8', errors='replace'))
        return "".join(parts)

    def save(self, path: str):
        with open(path, 'wb') as f:
            # Write big-endian int32
            f.write(struct.pack('>I', self.vocab_size))
            f.write(struct.pack('>I', len(self.merges)))
            for left, right in self.merges:
                f.write(struct.pack('>I', left))
                f.write(struct.pack('>I', right))

    def load(self, path: str):
        with open(path, 'rb') as f:
            target_vocab_size = struct.unpack('>I', f.read(4))[0]
            num_merges = struct.unpack('>I', f.read(4))[0]

            self.id_to_token.clear()
            self.token_to_id.clear()
            self.merges.clear()
            self.encode_cache.clear()

            # Base ASCII
            for i in range(256):
                s = bytes([i])
                self.id_to_token[i] = s
                self.token_to_id[s] = i
            self.id_to_token[self.UNK_ID] = b"<UNK>"
            self.token_to_id[b"<UNK>"] = self.UNK_ID

            for i in range(num_merges):
                left = struct.unpack('>I', f.read(4))[0]
                right = struct.unpack('>I', f.read(4))[0]
                new_token_id = 257 + i
                self.merges.append((left, right))

                t1 = self.id_to_token.get(left, b'')
                t2 = self.id_to_token.get(right, b'')
                new_token_str = t1 + t2
                self.id_to_token[new_token_id] = new_token_str
                self.token_to_id[new_token_str] = new_token_id

            self.vocab_size = target_vocab_size
