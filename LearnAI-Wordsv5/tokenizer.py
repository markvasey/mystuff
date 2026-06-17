import os
import json
import struct
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self):
        # Base byte-to-unicode character mapping used by GPT-2/LLaMA
        self.gpt2_map = self._bytes_to_unicode()
        self.unicode_to_byte = {c: b for b, c in self.gpt2_map.items()}
        self.sorted_alphabet = [self.gpt2_map[b] for b in range(256)]
        self.vocab = {char: i for i, char in enumerate(self.sorted_alphabet)}
        
        # Initialize the underlying Tokenizer with base bytes mapping to IDs 0-255
        self.tokenizer = Tokenizer(BPE(vocab=self.vocab, merges=[]))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        self.tokenizer.decoder = ByteLevelDecoder()
        self.vocab_size = 256

    def _bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def train(self, corpus: str, target_vocab_size: int):
        print(f"Training Rust-Backed BPE Tokenizer (Target Vocab: {target_vocab_size})...")
        
        # Hugging Face BpeTrainer to learn BPE merges up to target_vocab_size
        trainer = BpeTrainer(vocab_size=target_vocab_size, special_tokens=[])
        
        # Chunk corpus into non-empty lines to speed up pre-tokenization in parallel threads
        chunks = [line for line in corpus.splitlines() if line.strip()]
        if not chunks:
            chunks = [corpus]
            
        self.tokenizer.train_from_iterator(chunks, trainer=trainer)
        
        # Extract learned merges and rebuild the model to align base IDs strictly to 0-255
        temp_json = "temp_tokenizer_train.json"
        self.tokenizer.save(temp_json)
        with open(temp_json, "r") as f:
            data = json.load(f)
        if os.path.exists(temp_json):
            os.remove(temp_json)
            
        hf_merges = data["model"]["merges"]
        hf_merges_tuples = [tuple(m) for m in hf_merges]
        
        # Reconstruct vocabulary map where base tokens are 0-255 and merges start at 256+
        aligned_vocab = {char: idx for char, idx in self.vocab.items()}
        for idx, (left, right) in enumerate(hf_merges_tuples):
            aligned_vocab[left + right] = 256 + idx
            
        # Initialize fresh aligned tokenizer
        self.tokenizer = Tokenizer(BPE(vocab=aligned_vocab, merges=hf_merges_tuples))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        self.tokenizer.decoder = ByteLevelDecoder()
        
        self.vocab_size = 256 + len(hf_merges_tuples)
        print(f"Tokenizer trained. Vocab size: {self.vocab_size} ({len(hf_merges_tuples)} merges)")

    def encode(self, text: str) -> list[int]:
        # If it's a huge string, split it into chunks to use multithreaded encode_batch in Rust
        if len(text) > 1024 * 1024:
            chunks = text.split("\n\n")
            chunks = [c for c in chunks if c]
            encodings = self.tokenizer.encode_batch(chunks)
            ids = []
            for enc in encodings:
                ids.extend(enc.ids)
            return ids
        else:
            return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        # 1. Extract the merges from the model
        temp_json = "temp_tokenizer_save.json"
        self.tokenizer.save(temp_json)
        with open(temp_json, "r") as f:
            data = json.load(f)
        if os.path.exists(temp_json):
            os.remove(temp_json)
            
        hf_merges = data["model"]["merges"]
        
        # 2. Map the merges back to raw byte IDs for the custom binary format
        vocab_map = {}
        converted_merges = []
        next_id = 256
        for left, right in hf_merges:
            new_token = left + right
            
            # Map left component
            if len(left) == 1:
                left_id = self.unicode_to_byte[left]
            else:
                left_id = vocab_map[left]
                
            # Map right component
            if len(right) == 1:
                right_id = self.unicode_to_byte[right]
            else:
                right_id = vocab_map[right]
                
            vocab_map[new_token] = next_id
            next_id += 1
            
            converted_merges.append((left_id, right_id))
            
        # 3. Save to custom binary format (so the Java application can load it directly)
        with open(path, "wb") as f:
            f.write(struct.pack('>I', self.vocab_size))
            f.write(struct.pack('>I', len(converted_merges)))
            for left, right in converted_merges:
                f.write(struct.pack('>I', left))
                f.write(struct.pack('>I', right))
                
        # 4. Save Hugging Face compatible JSON configuration alongside it
        hf_json_path = path + ".json"
        self.tokenizer.save(hf_json_path)
        print(f"Saved binary tokenizer to {path}")
        print(f"Saved HF JSON tokenizer to {hf_json_path}")

    def load(self, path: str):
        hf_json_path = path + ".json"
        if os.path.exists(hf_json_path):
            # Load the optimized JSON configuration directly
            self.tokenizer = Tokenizer.from_file(hf_json_path)
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
            self.tokenizer.decoder = ByteLevelDecoder()
            self.vocab_size = self.tokenizer.get_vocab_size()
            print(f"Loaded HF JSON Tokenizer from {hf_json_path} (vocab size: {self.vocab_size})")
        elif os.path.exists(path):
            # Reconstruct the tokenizer dynamically from the custom binary format if JSON is missing
            print(f"HF JSON not found. Reconstructing Tokenizer from binary file {path}...")
            with open(path, 'rb') as f:
                target_vocab_size = struct.unpack('>I', f.read(4))[0]
                num_merges = struct.unpack('>I', f.read(4))[0]
                
                merges = []
                for _ in range(num_merges):
                    left = struct.unpack('>I', f.read(4))[0]
                    right = struct.unpack('>I', f.read(4))[0]
                    merges.append((left, right))
            
            # Map byte ID merges back to GPT-2 unicode tokens
            id_to_token = {i: self.gpt2_map[i] for i in range(256)}
            hf_merges_tuples = []
            
            for i, (left, right) in enumerate(merges):
                t1 = id_to_token[left]
                t2 = id_to_token[right]
                new_token_str = t1 + t2
                new_token_id = 256 + i
                id_to_token[new_token_id] = new_token_str
                hf_merges_tuples.append((t1, t2))
                
            aligned_vocab = {token: idx for idx, token in id_to_token.items()}
            
            self.tokenizer = Tokenizer(BPE(vocab=aligned_vocab, merges=hf_merges_tuples))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
            self.tokenizer.decoder = ByteLevelDecoder()
            self.vocab_size = target_vocab_size
            print(f"Reconstructed HF Tokenizer from binary. (vocab size: {self.vocab_size})")
        else:
            raise FileNotFoundError(f"Tokenizer file not found at {path}")
