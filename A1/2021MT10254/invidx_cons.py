import sys
from collections import Counter, defaultdict
import time
import dict_cons
import utils

fields_for_indexing = ['title', 'abstract']

def split_word(word, vocabulary, tokenizer_type):
    tokens = []
    if tokenizer_type != 2:
        i = 0
        while i < len(word):
            match_length = 0
            for length in range(1, len(word) - i + 1):
                if word[i:i + length] in vocabulary:
                    match_length = length
            tokens.append(word[i:i + match_length])
            i += match_length
    else:
        i = 0
        begin = 1
        while i < len(word):
            match_length = 0
            for length in range(1, len(word) - i + 1):
                if begin and word[i:i + length] in vocabulary:
                    match_length = length
                    continue
                if not begin and '##'+word[i:i + length] in vocabulary:
                    match_length = length
            tokens.append(word[i:i + match_length])
            i += match_length
            begin = 0
    return tokens

def get_tokenized_text(text, vocabulary, tokenizer_type=0):
    lst = []
    if not tokenizer_type:
        split_text =  utils.split_string_delimiters(text, dict_cons.delimiters)
    else:
        split_text =  utils.split_string_delimiters(text, dict_cons.delimiters_more)
    for word in split_text:
        lst.extend(split_word(word, vocabulary, tokenizer_type))
    return lst

def cons_inv_idx(term_dict, name_index, doc_ids):
    final_dict = {term: [len(doc_dict), list(doc_dict.items())] for term, doc_dict in term_dict.items()}
    utils.store_dictionary_with_offsets(final_dict, name_index + '.idx', name_index + '.dict', doc_ids)

def process_docs(docs_list, vocabulary, tokenizer_type, name_index):
    term_dict = defaultdict(Counter)
    i = 1
    t = time.time()
    
    doc_ids = []
    for doc in docs_list:
        doc_id = doc.get('doc_id')
        doc_ids.append(doc_id)
        text = ' '.join((doc[field] if field == 'title' else doc[field]) 
                        for field in fields_for_indexing if field in doc)
        for term in get_tokenized_text(text, vocabulary, tokenizer_type):
            term_dict[term][doc_id] += 1
        if i % 10000 == 0:
            t = time.time()
        i += 1
    cons_inv_idx(term_dict, name_index, doc_ids)

def main():
    coll_path = sys.argv[1]
    name_index = sys.argv[2]
    tokenizer_type = int(sys.argv[3])
    vocabulary = utils.load_vocab_as_set('output.dict')

    docs_list = utils.get_docs_list(coll_path)
    process_docs(docs_list, vocabulary, tokenizer_type, name_index)

if __name__ == '__main__':
    main()
