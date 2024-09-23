import re
import json
import os

STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
 "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
 "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
 "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", 
 "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", 
 "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", 
 "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", 
 "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", 
 "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", 
 "don", "should", "now"]
DELIMITERS = [' ', ',', '.', ':', ';', '"', "'"]

def split_string_delimiters(s, delimiters=DELIMITERS):
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

def merge_docs(docs_list, fields):
    return ' '.join((doc[field] if field == 'title' else doc[field]) 
                    for doc in docs_list for field in fields if field in doc)

def get_docs_list(path, type='docs'):
    return [json.loads(row) for row in (line for f in os.listdir(path) 
                                            if f.endswith(type) for line in open(os.path.join(path, f)))]



def write_results_to_file(results, header, file_path):
    with open(file_path, 'w') as file:
        if header:
            file.write(header + '\n')
        for result in results:
            file.write("{}\t{}\t{:.6f}\n".format(result[1], result[2], result[3]))

def process_text(text):
    text_result = []
    for word in split_string_delimiters(text, DELIMITERS):
        if word not in STOPWORDS:
            text_result.append(word)
    return ' '.join(text_result)


def get_docs_dict_from_queries(query_docs_dict, coll_path):
    docs_file = open(coll_path, 'r')
    docs_required = []
    for _, tup in query_docs_dict.items():
        docs_required.extend(tup[1])
    docs_required = set(docs_required)

    docs_dict = {}
    for row in docs_file:
        first_tab_pos = row.find('\t')
        doc_id = row[:first_tab_pos]
        if doc_id in docs_required:
                row_split = row.split('\t', 4)
                docs_dict[doc_id] = "{} {}".format(row_split[2], row_split[3])
    return docs_dict

def get_query_results(query_path, top100_path):
    top100_docs_file = open(top100_path, 'r')
    query_file = open(query_path, 'r')
    
    index_dict = {}
    next(query_file)
    for row in query_file:
        row = row.strip().split('\t')
        index_dict[int(row[0])] = (row[1], [])

    next(top100_docs_file)
    for row in top100_docs_file:
        row = row.strip().split('\t')
        index_dict[int(row[0])][1].append(row[1])
    return index_dict

def save_map_to_file(coll_freq_map, path):
    with open(path, 'w') as file:
        for word, freq in coll_freq_map.items():
            file.write("{}:{}\n".format(word, freq))

def write_words_to_file(words, file_path):
    with open(file_path, 'w') as file:
        for word in words:
            file.write("{}\n".format(word))

def write_results_to_file(results, path):
    with open(path, 'w') as file:
        for [score, query_id, doc_id, rank] in results:
            file.write("{}\tQ0\t{}\t{}\t{:.2f}\trunid1\n".format(query_id, doc_id, rank, score))

def write_expansions_to_file(query_id, expansions, expansions_file):
    expansions_str = ', '.join(expansions)
    expansions_file.write("{} : {}\n".format(query_id, expansions_str))

def get_map_from_file(path):
    coll_freq_map = {}
    with open(path, 'r') as file:
        for line in file:
            word, freq = line.strip().split('\t')
            coll_freq_map[word] = int(freq)
    return coll_freq_map

def get_list_from_file(path):
    with open(path, 'r') as file:
        words = [line.strip() for line in file]
    return words
