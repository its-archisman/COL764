import sys
from collections import Counter, defaultdict
from time import time
import utils

delimiters = [' ', ',', '.', ':', ';', '"', "'"]
delimiters_more = ['.', ' ', ':', ';', '"', "'", '.', '?', '!', ',', '\n', '~', '`', '(', ')', '/', '#', '*', '%', '+', '-', '[', ']', '{', '}', '@', '^']
vocab_size_bpe, num_merges_bpe, min_freq_bpe = 10000000, 20000, 2
vocab_size_wp, num_merges_wp, min_freq_wp = 10000000, 20000, 2


def simple_tokenizer(text):
    split_string_set = set(utils.split_string_delimiters(text, delimiters))
    vocab = sorted(list(split_string_set), key=lambda x: len(x), reverse=True)
    return vocab

def bpe_tokenizer(text):
    vocab_size, num_merges, min_freq = vocab_size_bpe, num_merges_bpe, min_freq_bpe
    tokens = utils.split_string_delimiters(text, delimiters_more)
    vocab = Counter(tokens)
    vocab = {(' '.join(word)): freq for word, freq in vocab.items()}

    final_vocab = set()
    for word in vocab:
        final_vocab.update(word.split())
    t = time()
    for _ in range(num_merges):
        pairs = utils.get_pair_stats(vocab)
        if not pairs:
            break
        most_common_pair = utils.get_most_common_pair(pairs, final_vocab)
        if pairs[most_common_pair] < min_freq:
            break
        vocab = utils.merge_pair_vocab(most_common_pair, vocab)
        final_vocab.add(utils.get_merged_pair(most_common_pair))
        if time() - t > 270:
            break
    
    final_vocab = sorted(final_vocab, key=lambda x: len(x), reverse=True)[:vocab_size]
    return final_vocab

def wordpiece_tokenizer(text):
    vocab_size, num_merges, min_freq = vocab_size_wp, num_merges_wp, min_freq_wp
    tokens = utils.split_string_delimiters(text, delimiters_more)
    vocab = Counter(tokens)
    vocab = {(' ##'.join(word)): freq for word, freq in vocab.items()}

    freq_indi = defaultdict()
    final_vocab = set()
    for word in vocab:
        final_vocab.update(word.split())
        for ch in word.split():
            if not ch in freq_indi:
                freq_indi[ch] = 0
            freq_indi[ch] += vocab[word]
    t = time()
    
    for _ in range(num_merges):
        pairs = utils.get_pair_stats(vocab)
        if not pairs:
            break
        wp_pair_scores = utils.get_wp_pair_scores(pairs, vocab, freq_indi)
        most_common_pair = utils.get_most_common_pair(wp_pair_scores, final_vocab, 1)
        vocab = utils.merge_pair_vocab(most_common_pair, vocab, 1)
        merged_str = utils.get_merged_pair(most_common_pair, 1)
        final_vocab.add(merged_str)
        freq_indi[merged_str] = pairs[most_common_pair]
        if time() - t > 270:
            break
    final_vocab = sorted(final_vocab, key=lambda x: len(x), reverse=True)[:vocab_size]
    return final_vocab

def main():
    coll_path = sys.argv[1]
    tokenizer_type = int(sys.argv[2])

    if tokenizer_type == 0:
        tokenizer = simple_tokenizer
    elif tokenizer_type == 1:
        tokenizer = bpe_tokenizer
    elif tokenizer_type == 2:
        tokenizer = wordpiece_tokenizer

    fields_for_vocab = ['title', 'abstract']
    # fields_for_vocab = ['title']
    import time
    t = time.time()
    docs_list = utils.get_docs_list(coll_path)
    n = len(docs_list)
    if tokenizer_type > 0:
        docs_list = docs_list[n//2:]
    combined_text = utils.merge_docs(docs_list, fields_for_vocab)
    vocab = tokenizer(combined_text)

    save_name = 'output.dict'
    utils.save_vocab(vocab, save_name)

if __name__ == '__main__':
    main()
