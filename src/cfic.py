from preprocessing import extract_sentences


class TrieNode:
    def __init__(self, position=-1):
        self.children = {}
        self.sentence_id = -1
        self.position = position

    def __repr__(self) -> str:
        rpr = (f"TrieNode(position={self.position}, "
               f"sentence_id={self.sentence_id}, "
               f"num_children={len(self.children)}, "
               f"children={self.children.keys()})")
        return rpr


class CFIC:
    def __init__(self, model, tokenizer, article, topk=3, max_length=256):
        self.model = model
        self.tokenizer = tokenizer
        self.topk = topk
        self.max_length = max_length
        self.trie_root = TrieNode()
        self.original_sentences = extract_sentences(article)
        self.sentence_tokens = []
        self.preprocess()

    def preprocess(self):
        for sent in self.original_sentences:
            self.sentence_tokens.append(self.tokenizer.tokenize(sent))

        for idx, sent in enumerate(self.sentence_tokens):
            self._insert_sentence_to_trie(sent, idx)

    def _insert_sentence_to_trie(self, tgt_sentence, tgt_sent_id):
        node = self._search_common_prefix(tgt_sentence)
        start_pos = node.position
        if node.sentence_id == -1:
            new_node = TrieNode(start_pos + 1)
            new_node.sentence_id = tgt_sent_id
            node.children[tgt_sentence[start_pos + 1]] = new_node
            return

        dup_sent_id = node.sentence_id
        dup_sentence = self.sentence_tokens[dup_sent_id]
        if (dup_sentence == tgt_sentence):
            return
        node.sentence_id = -1
        for i in range(start_pos + 1, min(len(tgt_sentence), len(dup_sentence))):
            t1 = tgt_sentence[i]
            t2 = dup_sentence[i]
            if t1 == t2:
                node.children[t1] = TrieNode(i)
                node = node.children[t1]
            else:
                node.children[t1] = TrieNode(i)
                node.children[t2] = TrieNode(i)
                node.children[t1].sentence_id = tgt_sent_id
                node.children[t2].sentence_id = dup_sent_id
                break

    def _search_common_prefix(self, sentence):
        curr = self.trie_root
        for token in sentence:
            if token in curr.children:
                curr = curr.children[token]
                if curr.sentence_id != -1:
                    return curr
            else:
                return curr
