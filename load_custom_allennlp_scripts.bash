# import allennlp
# print(allennlp.__file__)  # That's ALLENNLP_PATH/__init__.py, update the next line:
ALLENNLP_PATH=

cp src/cached_transformers.py ${ALLENNLP_PATH}/common/cached_transformers.py
cp src/pretrained_transformer_embedder.py ${ALLENNLP_PATH}/modules/token_embedders/pretrained_transformer_embedder.py
cp src/pretrained_transformer_indexer.py ${ALLENNLP_PATH}/data/token_indexers/pretrained_transformer_indexer.py
cp src/pretrained_transformer_mismatched_embedder.py ${ALLENNLP_PATH}/modules/token_embedders/pretrained_transformer_mismatched_embedder.py
cp src/pretrained_transformer_mismatched_indexer.py ${ALLENNLP_PATH}/data/token_indexers/pretrained_transformer_mismatched_indexer.py
cp src/pretrained_transformer_tokenizer.py ${ALLENNLP_PATH}/data/tokenizers/pretrained_transformer_tokenizer.py
cp src/text_field.py ${ALLENNLP_PATH}/data/fields/text_field.py
cp src/token_class.py ${ALLENNLP_PATH}/data/tokenizers/token_class.py