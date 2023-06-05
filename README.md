## Light Coreference Resolution for Russian with Hierarchical Discourse Features

This is our solution for [RuCoCo-23 shared task](https://www.dialog-21.ru/en/evaluation/2023/rucoco/): Coreference Resolution in Russian (only single antecedent resolution).

### 1. Set up the syntax & NER parser

- <details> <summary> <b>(Option 1) With Docker</b> </summary>
  
   * Run the container locally or remotely using the following command:
      ```commandline
         docker run --rm -d -p 3334:3333 --name spacy_ru tchewik/isanlp_spacy:ru
      ```   
   * Connect to it from Python: 
     ```python
     from isanlp.processor_remote import ProcessorRemote
      
     spacy_address = ['0.0.0.0', 3334]
     spacy_processor = (ProcessorRemote(spacy_address[0], spacy_address[1], '0'),
                        ['tokens', 'sentences'],
                        {'lemma': 'lemma',
                         'postag': 'postag',
                         'morph': 'morph',
                         'syntax_dep_tree': 'syntax_dep_tree',
                         'entities': 'entities'})
     ```
  </details>

- <details> <summary> <b>(Option 2) Locally</b> </summary>
  
   * Download the model
      ```commandline
     python -m spacy download ru_core_news_lg
      ``` 
   * Initialize in Python using [`ProcessorSpaCy`](https://github.com/IINemo/isanlp/blob/master/src/isanlp/processor_spacy.py)
      ```python
     from isanlp.processor_spacy import ProcessorSpaCy

     spacy_processor = (ProcessorSpaCy(model_name='ru_core_news_lg'),
                        ['tokens', 'sentences'],
                        {'lemma': 'lemma',
                         'postag': 'postag',
                         'morph': 'morph',
                         'syntax_dep_tree': 'syntax_dep_tree',
                         'entities': 'entities'})
      ```

### 2. Set up the RST parser (only for `model_rh`)

- <details> <summary> <b>(Only option) With Docker</b> </summary>
  
   * Run the container locally or remotely using the following command:
      ```commandline
      docker run --rm -d -p 3335:3333 --name rst_ru tchewik/isanlp_rst:2.1-rstreebank
      ```   
   * Connect to it from Python: 
     ```python
     from isanlp.processor_remote import ProcessorRemote
     
     rst_address = ['0.0.0.0', 3335]
     rst_processor = (ProcessorRemote(rst_address[0], rst_address[1], 'default'),
                      ['text', 'tokens', 'sentences', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
                      {'rst': 'rst'})
     ```
   
  </details>

### 3. Set up the coreference resolver

There are two models from the test leaderboard of RuCoCo-23: base and Rh-enhanced. The latter requires RST parsing which makes it slow. There are also two options for running: with Docker or locally.

| name    | F1 (dev) | F1 (test) | time (example, <br/>CPU only) | for local run <br/> (place into `models/`)                                     | docker image           |
|---------|----------|-----------|-------------------------------|--------------------------------------------------------------------------------|------------------------|
| base    | 74.3     | 72.8      | ~883 ms                       | [model_base.tar.gz](https://1drv.ms/u/s!AtBVo9P3Lsqihk5asX-XcK0s1CP8?e=vpZm9B) | `tchewik/corefhd:base` |
| base+rh | 74.6     | 73.3      | ~19 s                         | [model_rh.tar.gz](https://1drv.ms/u/s!AtBVo9P3Lsqihk3P5QZfn44v2ldJ?e=EybhCL)   | `tchewik/corefhd:rh`   |


- <details> <summary> <b>(Option 1) With Docker</b> </summary>
  
   * Run the [container](https://hub.docker.com/r/tchewik/isanlp_corefhd) locally or remotely using the following command using selected tag (`base` or `rh`):
      ```commandline
         docker run --rm -d -p 3336:3333 --name corefhd tchewik/isanlp_corefhd:<tag>
      ```   
   * Connect to it from Python: 
     ```python
     from isanlp.processor_remote import ProcessorRemote
     
     coref_address = ['0.0.0.0', 3335]
     
     # Base model
     corefhd = (ProcessorRemote(coref_address[0], coref_address[1], 'default'),
                ['text', 'tokens', 'sentences',
                 'lemma', 'postag', 'syntax_dep_tree', 'entities'],
                {'entity_clusters': 'entity_clusters'})
     
     # Rh model
     corefhd = (ProcessorRemote(coref_address[0], coref_address[1], 'default'),
                ['text', 'tokens', 'sentences',
                 'lemma', 'postag', 'syntax_dep_tree', 'entities', 'rst'],
                {'entity_clusters': 'entity_clusters'})
     ```
   
  </details>

- <details> <summary> <b>(Option 2) Locally</b> </summary>
  
  * Download the model as `models/model_base.tar.gz` or `models/model_rh.tar.gz` (link in the table).
  * Find the python path for allennlp and update for LUKE (see [`load_custom_allennlp_scripts.bash`](load_custom_allennlp_scripts.bash))
  * Initialize in Python using [`ProcessorCorefHD`](processor_corefhd.py):
     ```python
     from processor_corefhd import ProcessorCorefHD

     # Base model
     corefhd_processor = (ProcessorCorefHD(cuda_device=-1, use_discourse=False),
                ['text', 'tokens', 'sentences',
                 'lemma', 'postag', 'syntax_dep_tree', 'entities'],
                {0: 'entity_clusters'})
    
     # Rh model
     corefhd_processor = (ProcessorCorefHD(cuda_device=-1, use_discourse=True),
                ['text', 'tokens', 'sentences',
                 'lemma', 'postag', 'syntax_dep_tree', 'entities', 'rst'],
                {'entity_clusters': 'entity_clusters'})
    ```
  </details>

### 4. Process the texts

   * Construct the pipeline from initialized processors:
     * For <b>base model</b>
        ```python
          from isanlp import PipelineCommon
          from isanlp.processor_razdel import ProcessorRazdel
     
          ppl = PipelineCommon([
             (ProcessorRazdel(), ['text'],
              {'tokens': 'tokens',
               'sentences': 'sentences'}),
             spacy_processor,
             corefhd_processor
          ])
        ```
       
      * For <b>Rh model</b>
        ```python
          from isanlp import PipelineCommon
          from isanlp.processor_razdel import ProcessorRazdel
     
          ppl = PipelineCommon([
             (ProcessorRazdel(), ['text'],
              {'tokens': 'tokens',
               'sentences': 'sentences'}),
             spacy_processor,
             rst_processor,
             corefhd_processor
          ])
        ```
   
* Run the constructed pipeline:
   ```python
   text = open('text_example.txt', 'r').read().strip()
   result = ppl(text)
   ```
  The result is given in token spans:
   ```python
      >>> result['entity_clusters']
      [[[0, 1], [7, 7], [19, 19], [103, 104], [126, 126]],
       [[23, 27], [30, 30]],
       [[68, 69], [72, 72]],
       [[78, 83], [132, 132]],
       [[44, 53], [138, 138], [152, 152]],
       [[133, 134], [140, 140], [149, 149]],
       [[89, 90], [142, 142]]]
   ```
  Example finding the corresponding text spans:
  ```python
  def print_coreference_clusters(text, tokens, entity_clusters):
     def mention_to_str(mention):
         return text[tokens[mention[0]].begin: tokens[mention[1]].end]
     for entity in entity_clusters:
         print(f'{mention_to_str(entity[0])} ::: {[mention_to_str(mention) for mention in entity[1:]]}')
     
  >>> print_coreference_clusters(result['text'], result['tokens'], result['entity_clusters'])
  Иоганн Шильтбергер ::: ['он', 'отрок', 'сам Иоганн', 'он']
  рыцаря по имени Леонгарт Рихартингер ::: ['его']
  венгерские крестоносцы ::: ['которым']
  24-летним сыном герцога Бургундии Жаном Бесстрашным ::: ['Жана']
  венгерский король и будущий император Священной Римской империи Сигизмунд I ::: ['Сигизмунда', 'Сигизмунд']
  бургундские рыцари ::: ['Они', 'им']
  турецкой армией ::: ['турок']
  ```
