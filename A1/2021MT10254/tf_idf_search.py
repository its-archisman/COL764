import sys
from collections import Counter, defaultdict
from time import time
import dict_cons
import utils
import numpy as np
import invidx_cons

doc_fields_for_search = ['title', 'abstract']
query_fields_for_search = ['title', 'description']

stopwords = [
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "isn't", "it", "its", "itself", "just",
    "ll", "m", "me", "might", "more", "most", "must", "my", "myself",
    "need", "no", "nor", "not", "now", "o", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "re", "s", "same", "shan't", "she", "should", "shouldn't",
    "so", "some", "such", "t", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "ve",
    "very", "was", "wasn't", "we", "were", "weren't", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with",
    "won't", "would", "y", "you", "your", "yours", "yourself", "yourselves"
]

# N=192509
def load_dictionary(dict_file_path):
    dictionary = {}
    with open(dict_file_path, 'r') as dict_file:
        for line in dict_file:
            term, offset = line.strip().split(':')
            dictionary[term] = int(offset)
    return dictionary

def retrieve_postings(term, dictionary, index_file, first_line_length):
    if term not in dictionary:
        return 0, {}
    offset = dictionary[term]
    
    index_file.seek(first_line_length + offset)

    postings_list = index_file.readline().strip().split(';')
    postings_list = [ele.split(':') for ele in postings_list]
    freq = postings_list[0]
    postings_list = postings_list[1:]
    postings_dict = {ele[0]: ele[1] for ele in postings_list}
    return int(freq[0]), postings_dict

def tf_ij(f_ij):
    if f_ij >= 1:
        return np.log(f_ij) + 1
    return 0

def idf_i(df_i, N):
    if df_i == 0:
        return 0
    return np.log(1 + N/df_i)

def vsm_eval(query_text, tokenized_query_text_set, doc_id, inverted_index, N):
    
    sum, sq_d, sq_q = 0, 0, 0

    for term in tokenized_query_text_set:
        if not (row := inverted_index.get(term)):
            continue
        df_i, postings = row[0], row[1]
        if not postings:
            continue
        f_ij = int(postings.get(doc_id, 0))
        w_ij = tf_ij(f_ij) * idf_i(df_i, N)
        w_iq = tf_ij(query_text.count(term)) * 1
        
        sum += w_ij * w_iq
        sq_d += w_ij * w_ij
        sq_q += w_iq * w_iq
    if sq_d * sq_q == 0:
        return 0
    return sum/(np.sqrt(sq_d * sq_q))

def get_inverted_index(index_dict_add, index_file):
    index_dictionary = load_dictionary(index_dict_add)
    inverted_index = defaultdict()

    first_line_length = len(index_file.readline())
    for term in index_dictionary:
        ret1, ret2 = retrieve_postings(term, index_dictionary, index_file, first_line_length)
        inverted_index[term] = [ret1, ret2]
    return inverted_index

def remove_stopwords(text):
    split_text = utils.split_string_delimiters(text, dict_cons.delimiters_more)
    return ' '.join([word for word in split_text if word not in stopwords])

def main():
    
    query_file_add = sys.argv[1]
    result_file_add = sys.argv[2]
    index_file_add = sys.argv[3]
    index_dict_add = sys.argv[4]

    docs_ids = utils.retrieve_docs_ids_from_index(index_file_add)
    

    queries = utils.get_queries_list(query_file_add, 'queries')
    index_file = open(index_file_add, 'r')

    inverted_index = get_inverted_index(index_dict_add, index_file)
    vocabulary = set([key for key in inverted_index])

    N = len(docs_ids)

    results = []
    t = time()
    for query in queries:
        lst = []
        query_text = ' '.join(((query[field] + ' ')*2 if field == 'title' else query[field]) 
                        for field in query_fields_for_search if field in query)
        query_text = remove_stopwords(query_text)
        tokenized_query_text = invidx_cons.get_tokenized_text(query_text, vocabulary, 1)
        for doc_id in docs_ids:
            val = vsm_eval(query_text, set(tokenized_query_text), doc_id, inverted_index, N)
            if val > 0:
                lst.append((round(val, 12), doc_id))
        lst.sort()
        lst.reverse()
        lst = lst[:100]
        results.extend([(query.get('query_id'), tup[1], tup[0]) for tup in lst])

    utils.write_results_to_file(results, result_file_add)

if __name__ == '__main__':
    main()