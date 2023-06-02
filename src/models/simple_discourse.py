from typing import List, Optional

import numpy as np
from isanlp.annotation import Token
from isanlp.annotation_rst import DiscourseUnit
from multiprocessing import Process, Queue
import networkx as nx


class SimpleDiscourseUnit:
    """ Information of DU containing only token offsets and nuclearity. """

    def __init__(self,
                 id: int,
                 start_tok: int,
                 end_tok: int,
                 left=None,
                 right=None,
                 nuclearity: Optional[str] = None):
        self.id = id
        self.start_tok, self.end_tok = start_tok, end_tok
        self.left, self.right = left, right
        self.nuclearity = nuclearity

    def contains(self, start, end):
        return self.start_tok <= start <= end <= self.end_tok


class SimpleDiscourseTree:
    """ Input a discourse unit forest, keep as a single document tree. """

    def __init__(self,
                 tokens: List[Token],
                 units: List[DiscourseUnit]):
        self.tree = self._construct_tree(tokens, units)
        self.edus = SimpleDiscourseTree._get_edus(self.tree)
        self.edu_ids = [edu.id for edu in self.edus]

    def _construct_tree(self,
                        tokens: List[Token],
                        units: List[DiscourseUnit]):
        simpleunits = [SimpleDiscourseTree.simplify(tokens, du) for du in units]
        return self._merge_units(simpleunits)

    def _merge_units(self, simpleunits):
        if len(simpleunits) == 1:
            return simpleunits[0]

        new_units = simpleunits[:]
        new_id = simpleunits[-1].id + 1
        for i in range(len(new_units) - 2, -1, -1):
            new_units[i] = SimpleDiscourseUnit(
                id=new_id,
                start_tok=simpleunits[i].start_tok,
                end_tok=new_units[i + 1].end_tok,
                left=simpleunits[i],
                right=new_units[i + 1],
                nuclearity='NN'
            )
            new_id += 1
        return new_units[i]

    @staticmethod
    def _get_edus(tree):
        if not tree.left:
            return [tree]

        return SimpleDiscourseTree._get_edus(tree.left) + SimpleDiscourseTree._get_edus(tree.right)

    @staticmethod
    def _get_dependency_edges(tree, return_nuclearity=False):
        if not tree:
            return []

        if not tree.left:
            return []

        source_edu = SimpleDiscourseTree._most_nuclear_edu(tree.left).id
        target_edu = SimpleDiscourseTree._most_nuclear_edu(tree.right).id
        if tree.nuclearity == 'SN':
            source_edu, target_edu = target_edu, source_edu

        current_edge = [source_edu, target_edu]
        if return_nuclearity:
            current_edge.append(tree.nuclearity[0])

        return [current_edge] \
            + SimpleDiscourseTree._get_dependency_edges(tree.left, return_nuclearity=return_nuclearity) \
            + SimpleDiscourseTree._get_dependency_edges(tree.right, return_nuclearity=return_nuclearity)

    @staticmethod
    def _most_nuclear_edu(tree):
        if not tree.left:
            return tree

        if tree.nuclearity == 'SN':
            return SimpleDiscourseTree._most_nuclear_edu(tree.right)
        else:
            return SimpleDiscourseTree._most_nuclear_edu(tree.left)

    @staticmethod
    def simplify(tokens: List[Token],
                 unit: DiscourseUnit):

        if not unit:
            return None

        start_tok, end_tok = SimpleDiscourseTree.convert_offsets(unit.start, unit.end, tokens=tokens)
        return SimpleDiscourseUnit(
            id=unit.id,
            start_tok=start_tok,
            end_tok=end_tok,
            left=SimpleDiscourseTree.simplify(tokens, unit.left),
            right=SimpleDiscourseTree.simplify(tokens, unit.right),
            nuclearity=unit.nuclearity if unit.relation != 'elementary' else None
        )

    @staticmethod
    def convert_offsets(start: int, end: int, tokens: List[Token]):
        """ Converts char offset to token offset """
        assert start >= tokens[0].begin and end <= tokens[-1].end

        start_tok, end_tok = None, None

        for i, tok in enumerate(tokens):
            if tok.begin >= start:
                start_tok = i
                break

        for i, tok in enumerate(tokens):
            if tok.end >= end:
                end_tok = i
                break

        return start_tok, end_tok

    def find_lowest_du(self, start: int, end: int):
        lowest_du = self._locate_lowest_id(self.tree, start, end)
        return lowest_du

    @staticmethod
    def _locate_lowest_id(tree, start: int, end: int, previous_value=None):
        # Not in the current DU
        if start < tree.start_tok or end > tree.end_tok:
            return previous_value

        if tree.left:
            left_id = SimpleDiscourseTree._locate_lowest_id(tree.left, start, end, previous_value=None)
            if left_id:
                return left_id

            return SimpleDiscourseTree._locate_lowest_id(tree.right, start, end, previous_value=tree.id)

        # Du is elementary
        return tree.id

    def get_du(self, id):
        return self.find_du_by_id(self.tree, id)

    def find_leftmost_edu(self, id):
        start_point = self.get_du(id)
        return SimpleDiscourseTree._leftmost_edu(start_point)

    def find_rightmost_edu(self, id):
        start_point = self.get_du(id)
        return SimpleDiscourseTree._rightmost_edu(start_point)

    def find_lca(self, id1, id2):
        du1, du2 = self.get_du(id1), self.get_du(id2)
        span1 = (du1.start_tok, du1.end_tok)
        span2 = (du2.start_tok, du2.end_tok)
        spans = sorted([span1, span2])
        return SimpleDiscourseTree._find_lca(self.tree, *spans, previous_value=self.tree)

    @staticmethod
    def find_du_by_id(tree, id):
        if tree.id == id:
            return tree

        if tree.left:
            left_child = SimpleDiscourseTree.find_du_by_id(tree.left, id)
            if left_child:
                return left_child

            right_child = SimpleDiscourseTree.find_du_by_id(tree.right, id)
            return right_child

    @staticmethod
    def _find_lca(tree, left_span, right_span, previous_value=None):
        if not tree.left:
            return previous_value

        if tree.left.end_tok < left_span[0] \
                or tree.left.start_tok > left_span[1] \
                or tree.right.end_tok < right_span[0] \
                or tree.right.start_tok > right_span[1]:
            return previous_value

        left_result = SimpleDiscourseTree._find_lca(tree.left, left_span, right_span, previous_value=None)
        if left_result:
            return left_result

        return SimpleDiscourseTree._find_lca(tree.right, left_span, right_span, previous_value=tree)

    @staticmethod
    def _leftmost_edu(tree):
        if not tree.left:
            return tree.id

        return SimpleDiscourseTree._leftmost_edu(tree.left)

    @staticmethod
    def _rightmost_edu(tree):
        if not tree.left:
            return tree.id

        return SimpleDiscourseTree._rightmost_edu(tree.right)

    @staticmethod
    def _depth_of_du(du_id, tree, previous_value=0):
        previous_value += 1

        if tree.id == du_id:
            return previous_value

        if not tree.left:
            return 0

        return SimpleDiscourseTree._depth_of_du(du_id, tree.left, previous_value=previous_value) +\
            SimpleDiscourseTree._depth_of_du(du_id, tree.right, previous_value=previous_value)


class DiscourseFeaturesExtractor:
    def __init__(self):
        self.available_features = {
            'D_Lin': self.linear_distance,
            'D_Rh': self.rhetorical_distance,
            'D_LCA': self.distance_to_lca
        }

    def __call__(self, feature_name, **kwargs):
        return self.available_features[feature_name](**kwargs)

    @staticmethod
    def linear_distance(mention_du_id, antecedent_du_id, tree, **kwargs):
        return tree.edu_ids.index(mention_du_id) - tree.edu_ids.index(antecedent_du_id)

    @staticmethod
    def rhetorical_distance(mention_du_id, antecedent_du_id, lca, **kwargs):
        if mention_du_id == antecedent_du_id:
            return 1.

        graph = DiscourseFeaturesExtractor.subtree_to_graph(lca, half_nn_weights=False)
        shortest_path = nx.shortest_path(graph, source=str(mention_du_id), target=str(antecedent_du_id))
        if shortest_path:
            return nx.classes.function.path_weight(graph, shortest_path, weight='weight')
        return 100.

    @staticmethod
    def distance_to_lca(mention_du_id, lca, **kwargs):
        return SimpleDiscourseTree._depth_of_du(mention_du_id, lca)

    @staticmethod
    def subtree_to_graph(subtree, half_nn_weights=False):
        """ Compute the weighted graph matrix from constituency RST subtree """
        # all_edus = SimpleDiscourseTree._get_edus(subtree)
        all_edges = SimpleDiscourseTree._get_dependency_edges(subtree, return_nuclearity=True)
        G = nx.DiGraph()
        for source, target, source_nuclearity in all_edges:
            weight = 1.
            if half_nn_weights and source_nuclearity == 'N':
                weight = 0.5

            G.add_edge(str(source), str(target), weight=weight)
            G.add_edge(str(target), str(source), weight=1.)

        return G
