build.sh: This is for downloading the external dependencies for the python program files
Command: `./build.sh`

dictcons.sh: This is for building the vocabulary froom the given corpus of documents using a tokenizer
Command: `dictcons.sh [coll-path] {0|1|2}`
This will create a file named `output.dict` in the curent directory

invidx.sh: This is for constructing the inverted index
Command: `invidx.sh [coll-path] [indexfile] {0|1|2}`
This will create files `output.dict`, `indexfile.dict` and `indexfile.idx` in the current directory

tf_idf_search.sh: This is for gettings the query-search results
Command:    `tf idf search.sh [queryfile] [resultfile] [indexfile] [dictfile]`
This will create a file `resultfile` in the current directory

Expected usage of commands in order:
- `./build.sh`
- `dictcons.sh [coll-path] {0|1|2}` (not compulsory)
- `invidx.sh [coll-path] [indexfile] {0|1|2}`
- `tf idf search.sh [queryfile] [resultfile] [indexfile] [dictfile]`

Source code files: `dict_cons.py`, `inv_idx_cons.py`, `tf_idf_search.py`, `utils.py`
Assignment report: `2021MT10254.pdf`
