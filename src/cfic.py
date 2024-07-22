import torch
from preprocessing import clean_text, article_to_sentences


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
        self.article = clean_text(article)
        self.topk = topk
        self.max_length = max_length
        self.trie_root = TrieNode()
        self.original_sentences = article_to_sentences(article)
        self.sentence_tokens = []
        self.preprocess()
        self.prompt_prefix = "Below is an article. Memorize the article and answer my question after the article.\nThe article begins.\n"
        # self.prompt_suffix_template = "\nNow the article ends.\nUsing only the exact sentences from the above article to answer my question.\nQuestion: {}"
        self.prompt_suffix_template = "\nNow the article ends.\nUsing only the exact sentences from the above article to answer the Question without other words.\nQuestion: {}"

    def preprocess(self):
        for sent in self.original_sentences:
            self.sentence_tokens.append(self.tokenizer(sent).input_ids[1:])

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

    def constrained_sentence_prefix_decoding(self, question):
        input_text = self.prompt_prefix + self.article + self.prompt_suffix_template.format(question)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            logits, past_key_values = self.model(input_ids).to_tuple()
        logits = logits[0, -1].cpu()
        candidates = list(self.trie_root.children.keys())
        top_tokens = self._constrained_topk(logits, candidates, self.topk)

        top_sent_index = []
        for token in top_tokens:
            node = self.trie_root
            curr_past_key_values = past_key_values
            while node.children[token].sentence_id == -1:
                node = node.children[token]
                with torch.no_grad():
                    curr_logits, curr_past_key_values = self.model(
                        torch.tensor([[token]], device="cuda"),
                        past_key_values=curr_past_key_values,
                        use_cache=True).to_tuple()
                curr_logits = curr_logits[0, -1].cpu()
                candidates = list(node.children.keys())
                token = self._constrained_topk(logits, candidates, topk=1)[0]

            top_sent_index.append(node.children[token].sentence_id)
        return top_sent_index

    def _constrained_topk(self, logits, candidate_ids, topk):
        candidate_ids = torch.tensor(candidate_ids)
        _, temp_index = torch.topk(logits[candidate_ids], topk)
        top_tokens = candidate_ids[temp_index]
        return top_tokens.tolist()

    def skip_decoding(self):
        pass
