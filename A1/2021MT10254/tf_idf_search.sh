QUERYF=$1
RESULT_F=$2
INDEX_F=$3
DICTF=$4

python3 tf_idf_search.py "$QUERYF" "$RESULT_F" "$INDEX_F" "$DICTF"
