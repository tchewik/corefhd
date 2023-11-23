import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Set

from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from allennlp.data.fields import (
    Field,
    ListField,
    TextField,
    SpanField,
    MetadataField,
    SequenceLabelField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

from .ontonotes import OntonotesSentence
from .simple_discourse import SimpleDiscourseTree
from .span_extractor import SpanExtractor


def make_coref_instance(
        sentences: List[OntonotesSentence],
        token_indexers: Dict[str, TokenIndexer],
        max_span_width: int = 20,
        gold_clusters: Optional[List[List[Tuple[int, int]]]] = None,
        wordpiece_modeling_tokenizer: PretrainedTransformerTokenizer = None,
        max_sentences: int = None,
        remove_singleton_clusters: bool = True,
        span_extraction_strategy: str = 'rule-based',
        nlp_annot: dict = None,
        rst_annotation_dir: str = None,
        span_extraction_edu_restricted: bool = True,
) -> Instance:
    """
    # Parameters
    sentences : `List[List[OntonotesSentence]]`, required.
        A list of lists representing the tokenized words and sentences in the document.
    token_indexers : `Dict[str, TokenIndexer]`
        This is used to index the words in the document.  See :class:`TokenIndexer`.
    max_span_width : `int`, optional.
        The maximum width of candidate spans to consider (for span_extraction_strategy=='all')
    gold_clusters : `Optional[List[List[Tuple[int, int]]]]`, optional (default = `None`)
        A list of all clusters in the document, represented as word spans with absolute indices
        in the entire document. Each cluster contains some number of spans, which can be nested
        and overlap. If there are exact matches between clusters, they will be resolved
        using `_canonicalize_clusters`.
    wordpiece_modeling_tokenizer: `PretrainedTransformerTokenizer`, optional (default = `None`)
        If not None, this dataset reader does subword tokenization using the supplied tokenizer
        and distribute the labels to the resulting wordpieces. All the modeling will be based on
        wordpieces. If this is set to `False` (default), the user is expected to use
        `PretrainedTransformerMismatchedIndexer` and `PretrainedTransformerMismatchedEmbedder`,
        and the modeling will be on the word-level.
    max_sentences: `int`, optional (default = `None`)
        The maximum number of sentences in each document to keep. By default keeps all sentences.
    remove_singleton_clusters : `bool`, optional (default = `True`)
        Some datasets contain clusters that are singletons (i.e. no coreferents). This option allows
        the removal of them.
    span_extraction_strategy : `str`, optional (default = `rule-based`)
        Using 'all' strategy, we'll select all the token spans with length <= max_span_width.
        Using 'rule-based' strategy, we'll select only the noun phrases using some rules.
    span_extraction_edu_restricted : `bool`, optional (default = `True`)
        From all the extracted spans select only those occuring inside a single EDU.
        Requires `rst_annotation_dir` parameter.
    # Returns
    An `Instance` containing the following `Fields`:
        text : `TextField`
            The text of the full document.
        spans : `ListField[SpanField]`
            A ListField containing the spans represented as `SpanFields`
            with respect to the document text.
        span_labels : `SequenceLabelField`, optional
            The id of the cluster which each possible span belongs to, or -1 if it does
                not belong to a cluster. As these labels have variable length (it depends on
                how many spans we are considering), we represent this a as a `SequenceLabelField`
                with respect to the spans `ListField`.
        rst : `IsaNLP.annotation_rst.DiscourseUnit`, optional
    """
    if max_sentences is not None and len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
        total_length = sum(len(sentence) for sentence in sentences)

        if gold_clusters is not None:
            new_gold_clusters = []

            for cluster in gold_clusters:
                new_cluster = []
                for mention in cluster:
                    if mention[1] < total_length:
                        new_cluster.append(mention)
                if new_cluster:
                    new_gold_clusters.append(new_cluster)

            gold_clusters = new_gold_clusters

    # Collect the paragraphs
    all_paragraphs_tokens = []
    current_paragraph_tokens = []
    all_paragraphs_entities = []
    current_paragraph_entities = []
    tok_offset = 0
    for sentence in sentences:
        if sentence.new_paragraph[0] and current_paragraph_tokens:
            all_paragraphs_tokens.append(current_paragraph_tokens)
            current_paragraph_tokens = []
            all_paragraphs_entities.append(current_paragraph_entities)
            current_paragraph_entities = []
            tok_offset = 0

        current_paragraph_tokens += [(text, ner) for text, ner in zip(sentence.words, sentence.named_entities)]
        current_paragraph_entities += [(tok_offset + start, tok_offset + end) for start, end in
                                       sentence.named_entities_idxs]
        tok_offset += len(sentence.words)

    all_paragraphs_tokens.append(current_paragraph_tokens)
    all_paragraphs_entities.append(current_paragraph_entities)

    if wordpiece_modeling_tokenizer is not None:
        results = [wordpiece_modeling_tokenizer.intra_word_tokenize(span) for span in all_paragraphs_tokens]
        paragraphed_tokens, offsets = [res[0] for res in results], [res[1] for res in results]
    else:
        paragraphed_tokens = [[Token(text=word, ent_type_=ner) for word, ner in paragraph]
                              for paragraph in all_paragraphs_tokens]

    text_fields = ListField([TextField(pt, token_indexers, entity_spans=es)
                             for pt, es in zip(paragraphed_tokens, all_paragraphs_entities)])  # for text encoding
    flat_paragraph_tokens = [token for paragraph in paragraphed_tokens for token in
                             paragraph]  # for coreference indices
    flat_text_field = TextField(flat_paragraph_tokens, token_indexers)

    cluster_dict = {}
    if gold_clusters is not None:
        gold_clusters = _canonicalize_clusters(gold_clusters)
        if remove_singleton_clusters:
            gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 1]

        if wordpiece_modeling_tokenizer is not None:
            for cluster in gold_clusters:
                for mention_id, mention in enumerate(cluster):
                    start = offsets[mention[0]][0]
                    end = offsets[mention[1]][1]
                    cluster[mention_id] = (start, end)

        for cluster_id, cluster in enumerate(gold_clusters):
            for mention in cluster:
                cluster_dict[tuple(mention)] = cluster_id

    spans: List[Field] = []
    span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

    if nlp_annot:
        rst = SimpleDiscourseTree(tokens=nlp_annot['tokens'], units=nlp_annot['rst'])
    elif rst_annotation_dir:
        filename = sentences[0].document_id
        nlp_annot = pickle.load(open(os.path.join(rst_annotation_dir, filename + '.pkl'), 'rb'))
        rst = SimpleDiscourseTree(tokens=nlp_annot['tokens'], units=nlp_annot['rst'])
    else:
        rst = None

    sentence_offset = 0
    if span_extraction_strategy == 'rule-based':
        extractor = SpanExtractor()
    for sentence in sentences:
        if span_extraction_strategy == 'all':
            extracted_spans = enumerate_spans(sentence.words, offset=sentence_offset, max_span_width=max_span_width)
        elif span_extraction_strategy == 'rule-based':
            extracted_spans = extractor(sentence, offset=sentence_offset)

        for ner_span in sentence.named_entities_idxs:
            if ner_span not in extracted_spans:
                extracted_spans.append(ner_span)

        span_extraction_edu_restricted = span_extraction_edu_restricted if rst_annotation_dir else False
        if span_extraction_edu_restricted:
            filtered_spans = []
            for span_candidate in extracted_spans:
                if not span_candidate in filtered_spans:
                    for edu in rst.edus:
                        start_tok, end_tok = span_candidate
                        if edu.start_tok <= start_tok and edu.end_tok >= end_tok:
                            filtered_spans.append(span_candidate)
                            break

            extracted_spans = filtered_spans

        extracted_spans = sorted(extracted_spans)

        for start, end in extracted_spans:
            if wordpiece_modeling_tokenizer is not None:
                start = offsets[start][0]
                end = offsets[end][1]

                # `enumerate_spans` uses word-level width limit; here we apply it to wordpieces
                # We have to do this check here because we use a span width embedding that has
                # only `max_span_width` entries, and since we are doing wordpiece
                # modeling, the span width embedding operates on wordpiece lengths. So a check
                # here is necessary or else we wouldn't know how many entries there would be.
                if (span_extraction_strategy == 'all') and (end - start + 1 > max_span_width):
                    continue
                # We also don't generate spans that contain special tokens
                if start < len(wordpiece_modeling_tokenizer.single_sequence_start_tokens):
                    continue
                if end >= len(flat_paragraph_tokens) - len(
                        wordpiece_modeling_tokenizer.single_sequence_end_tokens
                ):
                    continue

            if span_labels is not None:
                if (start, end) in cluster_dict:
                    span_labels.append(cluster_dict[(start, end)])
                else:
                    span_labels.append(-1)

            spans.append(SpanField(start, end, flat_text_field))

        sentence_offset += len(sentence.words)

    span_field = ListField(spans)

    metadata: Dict[str, Any] = {
        "doc_id": sentences[0].document_id,
    }

    if gold_clusters is not None:
        metadata["clusters"] = gold_clusters
    metadata_field = MetadataField(metadata)

    fields: Dict[str, Field] = {
        "text": text_fields,
        "spans": span_field,
        "metadata": metadata_field,
    }

    if rst is not None:
        fields["rst"] = MetadataField(rst)

    if span_labels is not None:
        fields["span_labels"] = SequenceLabelField(span_labels, span_field)

    return Instance(fields)


def _normalize_word(word):
    if word in ("/.", "/?"):
        return word[1:]
    else:
        return word


def _canonicalize_clusters(clusters: List[List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The data might include 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters:
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]
