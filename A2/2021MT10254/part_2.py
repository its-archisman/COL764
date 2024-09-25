import sys
import os
from collections import Counter, defaultdict
from time import time
import numpy as np

import utils
import part_0 as rerank_utils

EXPANSIONS = 10

models_dir = 'models'
lm_path = models_dir + '/qt_model'

def get_vocab_and_embedding(file_path, model_type):
    vocab = []
    embedding_dict = {}
    
    file = open(file_path, 'r')
    if model_type == 'w2v':
        file.readline()

    for row in file:
        row_split = row.split()
        if not row_split:
            continue
        vocab.append(row_split[0])
        embedding = [float(st) for st in row_split[1:]]
        embedding_dict[row_split[0]] = embedding
    
    return vocab, embedding_dict


def get_expanded_query_pretrained(query_id, query_tuple, required_docs_dict, expansions_file, embedding_path, model_type):
    query_text = query_tuple[0]
    docs_retrieved = query_tuple[1]

    model_path = embedding_path

    combined_docs_text = ' '.join([required_docs_dict[key] for key in docs_retrieved])
    combined_docs_text = utils.process_text(combined_docs_text)

    vocab, embeddings_dict = get_vocab_and_embedding(model_path, model_type)
    embeddings_list = [embeddings_dict[term] for term in vocab]

    query_split = utils.split_string_delimiters(query_text)
    q = np.array([[1] if term in query_split else [0] for term in vocab])
    U = np.array(embeddings_list)
    vector = np.dot(U, np.dot(U.T, q)).flatten()
    sorted_indices = np.argsort(vector)
    top_indices = [i for i in sorted_indices if vocab[i] not in query_split][-EXPANSIONS:][::-1]

    word_expansions = [vocab[i] for i in top_indices]
    utils.write_expansions_to_file(query_id, word_expansions, expansions_file)
    
    query_text_new = query_text + ' ' + ' '.join(word_expansions)
    return query_text_new

def main():
    print("Running")
    query_path = sys.argv[1]
    top100_path = sys.argv[2]
    coll_path = sys.argv[3]
    embedding_path = sys.argv[4]
    out_file_path = sys.argv[5]
    expansions_file_path = sys.argv[6]
    model_type = sys.argv[7]

    expansions_file = open(expansions_file_path, 'w')
    if not os.path.isfile(lm_path):
        rerank_utils.train_query_translation_model(query_path, top100_path, coll_path, lm_path)

    query_docs_dict = utils.get_query_results(query_path, top100_path)
    required_docs_dict = utils.get_docs_dict_from_queries(query_docs_dict, coll_path)

    for query_id in query_docs_dict:
        query_text_new = get_expanded_query_pretrained(query_id, query_docs_dict[query_id], required_docs_dict, 
                                            expansions_file, embedding_path, model_type)
        query_docs_dict[query_id][0] = query_text_new
    reranked_results = rerank_utils.get_reranked_results(query_docs_dict, required_docs_dict, lm_path)
    utils.write_results_to_file(reranked_results, out_file_path)

    expansions_file.close()
    print("Done")

if __name__ == '__main__':
    main()
