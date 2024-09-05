import os
import json
import re
from collections import Counter, defaultdict


def split_string_delimiters(s, delimiters):
    result = []
    temp = []
    english_pattern = re.compile(r'[a-zA-Z]')
    for char in s:
        if char in delimiters:
            if temp:
                result.append(''.join(temp))
                temp = []
            continue
        elif english_pattern.match(char):
            if char != '_':
                char = char.lower()
            temp.append(char)
    if temp:
        result.append(''.join(temp))
    return result

def get_pair_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair_vocab(pair, vocab, type=0):
    new_vocab = {}
    for word in vocab:
        new_word = replace_str(word, pair, type)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def replace_str(st, pair, type=0):
    lst = st.split()
    new_lst = []
    i = 0
    flag = 0
    while i < len(lst)-1:
        if lst[i] == pair[0] and lst[i+1] == pair[1]:
            new_lst.append(get_merged_pair(pair, type))
            if i == len(lst) - 2:
                flag = 1
            i += 2
        else:
            new_lst.append(lst[i])
            i += 1
    if not flag:
        new_lst.append(lst[-1])
    return ' '.join(new_lst)

def get_string_freq_wp(vocab, st):
    return vocab.get(st, 1)

def get_merged_pair(pair, type=0):
    if type and pair[1].startswith('##'):
        merged_str = pair[0] + pair[1][2:]
    else:
        merged_str = ''.join(pair)
    return merged_str

def get_wp_pair_scores(pairs, vocab, freq_indi):
    pair_scores = defaultdict()
    for pair in pairs:
        pair_scores[pair] = pairs[pair]/(freq_indi[pair[0]] * freq_indi[pair[1]])
    return pair_scores

def get_most_common_pair(pair_scores, final_vocab, type=0):
    max_pair = None
    m = 0
    for pair in pair_scores:
        if not max_pair:
            max_pair = pair
        if get_merged_pair(pair, type) in final_vocab:
            continue
        if m < pair_scores[pair]:
            max_pair = pair
            m = pair_scores[pair]
    return max_pair

def merge_docs(docs_list, fields):
    return ' '.join((doc[field] if field == 'title' else doc[field]) 
                    for doc in docs_list for field in fields if field in doc)

def get_docs_list(path, type='docs'):
    return [json.loads(row) for row in (line for f in os.listdir(path) 
                                            if f.endswith(type) for line in open(os.path.join(path, f)))]

def get_queries_list(path, type='docs'):
    return [json.loads(row) for row in (line for line in open(path, 'r'))]

def save_vocab(vocab, output_file):
    with open(output_file, 'w') as file:
        for term in vocab:
            file.write(f"{term}\n")

def save_dict_to_file(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)

def load_vocab_as_set(filename):
    vocab = set()
    with open(filename, 'r') as file:
        for line in file:
            word = line.strip()
            if word:
                vocab.add(word)
    return vocab

def store_dictionary_with_offsets(dictionary, index_file_path, dict_file_path, doc_ids):
    offsets = {}
    flag = 1
    with open(index_file_path, 'w') as index_file:
        index_file.write(';'.join(doc_ids) + '\n')
        current_offset = 0
        for term, (doc_freq, postings) in dictionary.items():
            offsets[term] = current_offset
            postings_str = ';'.join([f"{tup[0]}:{tup[1]}" for tup in postings])
            index_file.write(f"{doc_freq};{postings_str}\n")
            current_offset += len(f"{doc_freq};{postings_str}") + 1  # +1 for the newline character

    with open(dict_file_path, 'w') as dict_file:
        for term, offset in offsets.items():
            dict_file.write(f"{term}:{offset}\n")

def cons_idx(pairs, name_index):
    doc_dict = defaultdict(lambda: defaultdict(int))
    for term, doc_id in pairs:
        doc_dict[doc_id][term] += 1
    final_dict = {doc_id: list(term_dict.items()) for doc_id, term_dict in doc_dict.items()}
    save_dict_to_file(final_dict, name_index + '.dict')

def transform_to_nested_dict(qrel_list):
    nested_dict = {}
    for entry in qrel_list:
        query_id = entry["query_id"]
        doc_id = entry["doc_id"]
        relevance = entry["relevance"]
        iteration = entry["iteration"]

        if query_id not in nested_dict:
            nested_dict[query_id] = {}

        nested_dict[query_id][doc_id] = {"relevance": relevance, "iteration": iteration}

    return nested_dict

def write_results_to_file(results, file):
    rows = []
    for result in results:
        rows.append([result[0], 0, result[1], result[2]])
    write_lists_to_file(file, rows, 'results')

def write_lists_to_file(file_path, list_of_lists, type='normal'):
    with open(file_path, 'w') as file:
        if type == 'results':
            file.write('\t'.join(map(str, ['qid', 'iteration', 'docid', 'relevancy'])) + '\n')
        for sublist in list_of_lists:
            file.write('\t'.join(map(str, sublist)) + '\n')

def retrieve_docs_ids_from_index(index_file_path):
    with open(index_file_path, 'r') as index_file:
        docs_list_line = index_file.readline().strip()
        docs_list = docs_list_line.split(';')

    return docs_list

def get_unique_docs_list(docs_list):
    docs_dict = defaultdict()
    for doc in docs_list:
        if not doc['title'] in docs_dict:
            docs_dict[doc['title']] = doc
    return [val for _, val in docs_dict.items()]
