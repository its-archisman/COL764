import sys
import os
from collections import Counter, defaultdict
from time import time
import utils
import numpy as np

import utils
import w2v
import part_0 as rerank_utils

EXPANSIONS = 10

models_dir = 'models'
lm_path = models_dir + '/qt_model'

def get_expanded_query(query_id, query_tuple, required_docs_dict, expansions_file):
    query_text = query_tuple[0]
    docs_retrieved = query_tuple[1]

    model_path = models_dir + '/local/' + str(query_id)
    vocab_path = models_dir + '/vocab/' + str(query_id)

    combined_docs_text = ' '.join([required_docs_dict[key] for key in docs_retrieved])
    combined_docs_text = utils.process_text(combined_docs_text)

    if not os.path.isfile(model_path) or not os.path.isfile(vocab_path):
        w2v.run(combined_docs_text, model_path, vocab_path)

    vocab = utils.get_list_from_file(vocab_path)

    sg = w2v.SkipGram.load(model_path)
    embeddings_list = [sg.get_embedding(term) for term in vocab]

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
    query_path = sys.argv[1]
    top100_path = sys.argv[2]
    coll_path = sys.argv[3]
    out_file_path = sys.argv[4]
    expansions_file_path = sys.argv[5]

    expansions_file = open(expansions_file_path, 'w')

    if not os.path.isfile(lm_path):
        rerank_utils.train_query_translation_model(query_path, top100_path, coll_path, lm_path)

    query_docs_dict = utils.get_query_results(query_path, top100_path)
    required_docs_dict = utils.get_docs_dict_from_queries(query_docs_dict, coll_path)
    for query_id in query_docs_dict:
        query_text_new = get_expanded_query(query_id, query_docs_dict[query_id], required_docs_dict, 
                                            expansions_file)
        query_docs_dict[query_id][0] = query_text_new

    reranked_results = rerank_utils.get_reranked_results(query_docs_dict, required_docs_dict, lm_path)
    utils.write_results_to_file(reranked_results, out_file_path)

    expansions_file.close()

if __name__ == '__main__':
    main()
