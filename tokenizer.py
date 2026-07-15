import json
import os
import re

class BPETokenizer:
    def __init__(self, merges_path):
        with open(merges_path, "r", encoding="utf-8") as f:
            self.merges_list = json.load(f)
        
        self.merges = {}
        for pair, new_id in self.merges_list:
            self.merges[tuple(pair)] = new_id
            
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, new_id in self.merges_list:
            p0, p1 = pair
            self.vocab[new_id] = self.vocab[p0] + self.vocab[p1]
            
        self.vocab_size = 256 + len(self.merges_list)
        self.cache = {}
        
    def _encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        
        word_bytes = list(word.encode("utf-8"))
        if len(word_bytes) <= 1:
            return word_bytes
            
        ids = word_bytes
        while len(ids) > 1:
            pairs = list(zip(ids[:-1], ids[1:]))
            best_pair = None
            best_id = float('inf')
            for pair in pairs:
                if pair in self.merges:
                    new_id = self.merges[pair]
                    if new_id < best_id:
                        best_id = new_id
                        best_pair = pair
            
            if best_pair is None:
                break
                
            new_ids = []
            skip = False
            for j in range(len(ids)):
                if skip:
                    skip = False
                    continue
                if j < len(ids) - 1 and (ids[j], ids[j+1]) == best_pair:
                    new_ids.append(best_id)
                    skip = True
                else:
                    new_ids.append(ids[j])
            ids = new_ids
            
        self.cache[word] = ids
        return ids

    def encode(self, text):
        words = re.findall(r'\s+|\S+', text)
        encoded_ids = []
        for word in words:
            encoded_ids.extend(self._encode_word(word))
        return encoded_ids

    def decode(self, ids):
        b = bytearray()
        for i in ids:
            if i in self.vocab:
                b.extend(self.vocab[i])
            else:
                b.append(i % 256)
        return b.decode("utf-8", errors="replace")


def load(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
    return BPETokenizer(path)
