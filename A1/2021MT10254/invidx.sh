COLL_PATH=$1
INDEX=$2
TOKENIZER=$3

python3 dict_cons.py "$COLL_PATH" "$TOKENIZER"
python3 invidx_cons.py "$COLL_PATH" "$INDEX" "$TOKENIZER"
