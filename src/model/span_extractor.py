""" Adapted rule-based span extractor from the RuCoCo-23 Shared Task baseline. """

from typing import Iterator, Tuple, List

from .ontonotes import OntonotesSentence


class SpanExtractor:
    skipped_pos = {"ADP", "CCONJ", "SCONJ"}
    allowed_punct = {"\"", "'", "(", ")", "."}

    def __call__(self, sentence: OntonotesSentence, offset=0) -> Iterator[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for i in range(len(sentence.words)):
            if sentence.pos_tags[i] in {'DET', 'PRON'}:
                spans.append((i, i))

            elif sentence.pos_tags[i] in {'NOUN', 'PROPN'}:
                start, end = i, i
                for j in range(i - 1, -1, -1):
                    if not SpanExtractor._is_ancestor(i, j, sentence.parse_tree):
                        break

                    if sentence.parse_tree[j] == i:
                        if self._is_participle_phrase(j, sentence.pos_tags, sentence.parse_tree):
                            break
                        if self._is_skipped_pos(j, sentence.words, sentence.pos_tags):
                            continue

                        leftmost = \
                        sorted([node for node in SpanExtractor._token_subtree(j, sentence.parse_tree) if node not in [i, -1]])[0]
                        if leftmost < start:
                            start = leftmost

                for j in range(i + 1, len(sentence.words)):
                    if not SpanExtractor._is_ancestor(i, j, sentence.parse_tree):
                        break

                    if sentence.parse_tree[j][0] == i:
                        if self._is_participle_phrase(j, sentence.pos_tags, sentence.parse_tree):
                            break
                        if self._is_skipped_pos(j, sentence.words, sentence.pos_tags):
                            continue

                        rightmost = \
                        sorted([node for node in SpanExtractor._token_subtree(j, sentence.parse_tree) if node != i and node not in [i, -1]])[-1]
                        if rightmost > end:
                            end = rightmost

                spans.append((start, end))

        spans = [(offset + span[0], offset + span[1]) for span in spans]
        return spans

    @staticmethod
    def _token_subtree(i: int, parse_tree: list) -> Iterator[int]:
        yield i
        for child in [j for j, (head, rel) in enumerate(parse_tree) if head == i]:
            yield from SpanExtractor._token_subtree(child, parse_tree)

    @staticmethod
    def _is_participle_phrase(i: int, pos_tags: list, parse_tree: list) -> bool:
        return pos_tags[i] == "VERB" and any(i != j for j in SpanExtractor._token_subtree(i, parse_tree))

    @staticmethod
    def _is_skipped_pos(i: int, words: list, pos_tags: list) -> bool:
        return pos_tags[i] in SpanExtractor.skipped_pos or (
                pos_tags[i] == "PUNCT" and words[i] not in SpanExtractor.allowed_punct)

    @staticmethod
    def _is_ancestor(i: int, j: int, parse_tree: list) -> bool:
        while j != i and j != -1:
            j = parse_tree[j][0]
        return j == i
