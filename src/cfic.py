import torch
import torch.nn.functional as F
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
        self.sentences = article_to_sentences(self.article)
        self.sent_tokens = []
        self._preprocess()
        self.prompt_prefix = "Below is an article. Memorize the article and answer my question after the article.\nThe article begins.\n"
        # self.prompt_suffix_template = "\nNow the article ends.\nUsing only the exact sentences from the above article to answer my question.\nQuestion: {}"
        self.prompt_suffix_template = "\nNow the article ends.\nUsing only the exact sentences from the above article to answer the Question without other words.\nQuestion: {}"

    def _preprocess(self):
        for sent in self.sentences:
            self.sent_tokens.append(self.tokenizer(sent).input_ids[1:])

        for idx, sent in enumerate(self.sent_tokens):
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
        dup_sentence = self.sent_tokens[dup_sent_id]
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

    def generate_grounding_texts(self, question):
        cspd_output = self.constrained_sentence_prefix_decoding(question)
        top_sent_start_indices, past_key_values = cspd_output

        gounding_texts = self.skip_decoding(top_sent_start_indices, past_key_values)
        return gounding_texts

    def constrained_sentence_prefix_decoding(self, question):
        input_text = self.prompt_prefix + self.article + self.prompt_suffix_template.format(question)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

        logits, past_key_values = self._compute_next_token_logits_with_cache(input_ids)
        logits = logits[0, -1].cpu()
        candidates = list(self.trie_root.children.keys())
        top_tokens = self._constrained_topk(logits, candidates, self.topk)

        top_sent_indices = []
        for token in top_tokens:
            node = self.trie_root
            curr_past_key_values = past_key_values
            while node.children[token].sentence_id == -1:
                node = node.children[token]
                curr_logits, curr_past_key_values = self._compute_next_token_logits_with_cache(
                    torch.tensor([[token]], device="cuda"), curr_past_key_values)
                curr_logits = curr_logits[0, -1].cpu()
                candidates = list(node.children.keys())
                token = self._constrained_topk(logits, candidates, topk=1)[0]

            top_sent_indices.append(node.children[token].sentence_id)
        return top_sent_indices, past_key_values

    @torch.no_grad()
    def _compute_next_token_logits_with_cache(self, input_ids, past_key_values=None):
        if past_key_values is None:
            output = self.model(input_ids)
        else:
            output = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        logits = output.logits
        past_key_values = output.past_key_values
        return logits, past_key_values

    def _constrained_topk(self, logits, candidate_ids, topk):
        candidate_ids = torch.tensor(candidate_ids)
        _, temp_index = torch.topk(logits[candidate_ids], topk)
        top_tokens = candidate_ids[temp_index]
        return top_tokens.tolist()

    def skip_decoding(self, top_start_sent_indices, past_key_values):
        top_end_sent_indices = []
        for sent_idx in top_start_sent_indices:
            curr_length = len(self.sent_tokens[sent_idx])
            curr_sent = self.sentences[sent_idx]
            curr_idx = sent_idx
            eos_probs, cand_end_indices = [], []
            while curr_idx < len(self.sentences) and curr_length <= self.max_length:
                input_ids = self.tokenizer(curr_sent, return_tensors="pt").input_ids.to("cuda")
                logits, _ = self._compute_next_token_logits_with_cache(input_ids, past_key_values)
                logits = logits[0, -1]
                prob = F.softmax(logits, dim=-1)[self.tokenizer.eos_token_id].item()
                eos_probs.append(prob)
                cand_end_indices.append(curr_idx)
                curr_idx += 1
                curr_length += len(self.sent_tokens[curr_idx])
                curr_sent += " " + self.sentences[curr_idx]
            if len(eos_probs) == 0:
                continue
            _, max_idx = torch.max(torch.tensor(eos_probs), dim=0)
            top_end_sent_indices.append(cand_end_indices[max_idx])

        nonoverlapping_intervals = self._merge_overlapping(
            top_start_sent_indices, top_end_sent_indices)

        self.grounding_texts = []
        for start, end in nonoverlapping_intervals:
            self.grounding_texts.append(" ".join(self.sentences[start:end+1]))
        return self.grounding_texts

    def _merge_overlapping(self, top_start_sent_indices, top_end_sent_indices):
        """leetcode 56"""
        intervals = list(zip(top_start_sent_indices, top_end_sent_indices))
        intervals.sort(key=lambda x: x[0])
        start, end = intervals[0]
        nonoverlapping_intervals = []
        for i in range(1, len(intervals)):
            if intervals[i][0] <= end:
                end = max(end, intervals[i][1])
            else:
                nonoverlapping_intervals.append((start, end))
                start, end = intervals[i]
        nonoverlapping_intervals.append((start, end))
        return nonoverlapping_intervals
