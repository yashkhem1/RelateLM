1. Use the create_pretraining_data_ENS.py in order to change the standard BERT sequence creation with the Empty-Next-Sentence Setting.
2. The ideally means that there is no sense of NSP Task. The sequences created are either of the form [CLS]Sentences[SEP][SEP] or [CLS][SEP]Sentences[SEP].
3. Use this directory in Tensorflow V2 setting