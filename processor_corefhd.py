import os

from src.model.light_coref_predictor import LightCorefPredictor


class ProcessorCorefHD:
    def __init__(self, use_discourse=False, model_dir='models/', cuda_device=-1):
        self._use_discourse = use_discourse
        model_name = 'model_rh.tar.gz' if self._use_discourse else 'model_base.tar.gz'
        self.model = LightCorefPredictor.from_path(os.path.join(model_dir, model_name),
                                                   predictor_name='light_coreference_resolution',
                                                   cuda_device=cuda_device)

    def __call__(self, annot_text, annot_tokens, annot_sentences,
                 annot_lemma, annot_postag, annot_syntax_dep_tree, annot_entities, annot_rst=None,
                 *args, **kwargs):
        if not self._use_discourse:
            annot_rst = None

        return self.model.predict_isanlp(annot_text, annot_tokens, annot_sentences,
                                         annot_lemma, annot_postag, annot_syntax_dep_tree, annot_entities,
                                         annot_rst).get('clusters', [])
