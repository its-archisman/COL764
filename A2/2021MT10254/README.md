## The files and directories included in this submission are as follows:
- `2021MT10254.pdf`: Assignment report
- `build.sh`: Empty file, which is supposed to build external dependencies

- `part0.py` | `part_1.py` | `part_2.py` | `w2v.py` | `utils.py`: Python scripts

- `w2v-local_rerank.sh`: Shell script used to run the command `w2v-local rerank.sh [query-file] [top-100-file] [collection-file] [output-file] [expansions-file]`
While it runs, it will learn and save models at `./models/local`, `./models/vocab`, `./models/qt_model`; generates `[output-file]` and `[expansions-file]` as outputs

- `w2v-gen_rerank.sh | glove-gen_rerank.sh`: Shell scripts used to run the command `[w2v | glove]-gen_rerank.sh [query-file] [top-100-file] [collection-file] [embeddings-file] [output-file] [expansions-file]` 
While it runs it will use `./models/qt_model`; generates `[output-file]` and `[expansions-file]` as outputs

- The `models` directory and its associated subdirectories is used to save models during (and after) training