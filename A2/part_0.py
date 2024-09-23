import sys
from collections import Counter, defaultdict
from time import time
import utils
import numpy as np

DELIMITERS = [' ', ',', '.', ':', ';', '"', "'"]
MEW_DIRICHLET = 1000

def tokenize_and_map(text):
    return Counter(utils.split_string_delimiters(text, DELIMITERS))

def cal_prob(term, coll_freq_map, doc_freq_map, corpus_size, doc_size):
    p_coll_term = coll_freq_map[term]/corpus_size
    freq_term_doc = doc_freq_map[term]
    return (freq_term_doc + MEW_DIRICHLET * p_coll_term)/(doc_size + MEW_DIRICHLET)

def get_query_doc_score(query_text, doc_text, coll_freq_map, corpus_size):
    split_query_text = utils.split_string_delimiters(query_text, DELIMITERS)
    doc_freq_map = tokenize_and_map(doc_text)
    doc_size = len(doc_text)
    score = 0
    for term in split_query_text:
        score += np.log(cal_prob(term, coll_freq_map, doc_freq_map, corpus_size, doc_size))
    return score

def get_reranked_results(query_docs_dict, required_docs_dict, lm_path):
    coll_freq_map = utils.get_map_from_file(lm_path)
    corpus_size = sum(val for _, val in coll_freq_map.items())
    
    results = []
    for query_id, tup in query_docs_dict.items():
        score_list = []
        for doc_id in tup[1]:
            doc_text = required_docs_dict[doc_id]
            score = get_query_doc_score(tup[0], doc_text, coll_freq_map, corpus_size)
            score_list.append([score, query_id, doc_id])
        score_list.sort()
        for i, score in enumerate(score_list):
            score.append(i + 1)
        results.extend(score_list)
    return results

def train_query_translation_model(query_path, top100_path, coll_path, model_path):
    query_docs_dict = utils.get_query_results(query_path, top100_path)
    required_docs_dict = utils.get_docs_dict_from_queries(query_docs_dict, coll_path)
    print("Number of documents", len(required_docs_dict))
    combined_docs_text = ' '.join([required_docs_dict[doc_id] for val in query_docs_dict.values() for doc_id in val[1]])
    coll_freq_map = tokenize_and_map(combined_docs_text)

    utils.save_map_to_file(coll_freq_map, model_path)
