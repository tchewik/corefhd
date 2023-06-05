from isanlp import PipelineCommon

from processor_corefhd import ProcessorCorefHD


def create_pipeline(delay_init):
    pipeline_default = PipelineCommon([(ProcessorCorefHD(use_discourse=False),
                                        ['text', 'tokens', 'sentences',
                                         'lemma', 'postag', 'syntax_dep_tree', 'entities'],
                                        {0: 'entity_clusters'})
                                       ],
                                      name='default')

    return pipeline_default
