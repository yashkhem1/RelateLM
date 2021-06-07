# RelateLM
This repository contains the source code of the following ACL-IJCNLP 2021 paper: [Exploiting Language Relatedness for Low Web-Resource Language Model Adaptation: An Indic Languages Study](https://www.example.com). It is divided into 5 sections:
- [Setup](#setup)
- [Pretraining with MLM](#pretraining-with-mlm)
- [Pretraining with Alignment Loss](#pretraining-with-alignment-loss)
- [Fine-Tuning on Downstream Tasks](#fine-tuning-on-downstream-tasks)
- [Miscellaneous](#miscellaneous)

## Setup
First, create the conda environment from [relatelm_env.yml](https://github.com/yashkhem1/RelateLM/blob/master/relatelm_env.yml) file using:
```shell
conda env create -f relatelm_env.yml
conda activate relatelm_env
```
After that, setup **indic-trans** library using the instructions from [this](https://github.com/libindic/indic-trans) repository.<br>
Also note that the pretraining has been done using Google Cloud TPUs so some of the code will be TPU-specific.
## Pretraining with MLM
We need to create 2 new conda environments for Pretraining with BERT. We will make use of some code from [Google BERT Repo](https://github.com/google-research/bert) along with our code. Pretraining BERT has 2 components:
1. Preprocessing: <br>
(a) The current BERT Preprocessig code needs to run in Tensorflow v2. Create a new conda environment and set it up as follows:
```shell
conda env create --name bert_preprocessing
conda activate bert_preprocessing
conda install tensorflow==2.3.0

```
(b) Run the following command from the directory "BERT Pretraining and Preprocessing/Preprocessing Code" to create the preprocessing code. Refer to the [Google BERT Repo](https://github.com/google-research/bert) for other information.

```shell
python3 create_pretraining_data_ENS.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --do_whole_word_mask=True \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

2. Pre-training:<br>
(a) The BERT Pretraining code used needs to run in Tensorflow v1 (same as the original Google BERT). Create a new conda environment and set it up as follows:
```shell
conda env create --name bert_pretraining
conda activate bert_pretraining
conda install -c conda-forge tensorflow==1.14

```
(b) Clone the Original [Google BERT Repo](https://github.com/google-research/bert) and replace the create_pretraining_data.py with our "BERT Pretraining and Preprocessing/Pretraining Diff Files/run_pretraining_without_NSP.py". Note that to run the pretraining on TPUs, the init_checkpoint, input_file and output_dir need to be on a Google Cloud Bucket.
Run the following command for pretraining:

```shell
python run_pretraining_without_NSP.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_CONFIG_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --save_checkpoint_steps=10 \
  --iterations_per_loop=5 \
  --use_tpu=True \
  --tpu_name=node-1 \
  --tpu_zone=zone-1 \
  --num_tpu_cores=8 
```


## Pretraining with Alignment Loss
This section is comprised of multiple sub-sections which include instructions from creating Bilingual Lexicons to the final training. The final training code requires access to Google Cloud TPUs, however it shouldn't be quite difficult to modify it to run on GPUs (which might take a long time though). A sample script including the entire pipeline is present [here](https://github.com/yashkhem1/RelateLM/blob/master/scripts/relatelm_gujarati.sh)
### Frequency files
We first need to create frequency file for a language, which is nothing but a pickle file containing the frequency counter of the words present in the monolingual data. To create the frequency file, run the following:
```shell
python3 statistics.py\
    --stats word_freq\
    --mono path_to_monolingual_data\
    --outfile path_to_output_pickle_file
```
`--mono` : Path to the monolingual (text) data for the language. Note that the monolingual data (wherever used) should have the sentences in separate lines and documents separated by a blank line (\n)<br>
`--outfile` : Path to output frequency pickle file

### Bilingual Lexicons
The second step is to create a bi-directional bilingual lexicon (dictionary) between the source and the target language. The file is a one-to-many mapping between words in source language and target language, along with the frequencies in which they appear in the monolingual data. In order to create such dictionary (in pickle format), run the following command
```shell
python3 create_bilingual_dictionary.py\
    --l1 lang_1\
    --l2 lang_2\
    --bilingual path_to_dir_1 [path_to_dir_2]\
    --o1 path_to_dictionary_pickle_lang1_lang2\
    --o2 path_to_dictionary_pickle_lang2_lang1\
    --freq_1 lang_1_freq_file\
    --freq_2 lang_2_freq_file\
    [--include_wiktionary]\
    [--transliterate --trans_lang l1/l2]\

```
`--l1` : Code for language 1 ("pan":Punjabi, "eng": English, "hin": Hindi, "ben": Bengali, "guj": Gujarati, "asm": Assamese, "ori": Oriya)<br>
`--l2` : Code for language 2<br>
`--bilingual` : Path to folders containing raw dictionary files (we downloaded from [CFILT](http://www.cfilt.iitb.ac.in/~sudha/bilingual_mapping.tar.gz) and [Wiktionary](https://hi.wiktionary.org/wiki/)). The first folder should have mappings from language 1 to language 2 and the second folder (if any) should have mappings from language 2 to language 1. Sample raw data folders are present [here](https://github.com/yashkhem1/RelateLM/blob/master/Bilingual_Lexicons)<br>
`--o1` : Path to output dictionary pickle file for the direction lang1 -> lang2<br>
`--o2` : Path to output dictionary pickle file for the direction lang2 -> lang1<br>
`--freq1` : Path to frequency file of language 1 (created above)<br>
`--freq2` : Path to frequency file of language 2 (created above)<br>
`--include_wiktionary` : Whether you wish to include Wiktionary files. If this is not used, then the files named *wiktionary.txt* in the folders will not be used.<br>
`--transliterate` : For transliterating the dictionary words<br>
`--trans_lang` : **l1** or **l2** depending on which language's script you wish  to transliterate the dictionary in <br>

### Pseudo Translations
Using the dictionary created above, we now create the pseudo translation file from the monolingual data. To do this, run the following command:
```shell
python3 create_pseudo_translation.py\
    --mono path_to_monolingual_data\
    --dict_path path_to_bilingual_lexicon_file\
    --outfile path_to_output_pseudo_translation_file\
    --replace replacement_method
```
`--mono` : Path to the monolingual (text) data<br>
`--dict_path` : Path to bilingual lexicon file (created above). Note that the lexicon file should be *from* the direction of tha language of monolingual data, and the keys should be in the same script as monolingual data.<br>
`--outfile` : Path to output pseudo translation (text) file. The output file will have the same number of lines and each line will have the same number of (whole) words as the monolingual data<br>
`--replace` : Method of choosing the word translation from the multiple possibilites. The methods applicable are:
- **problin**: Probabilistically sample the translations based on their frequencies ("weighted" in the paper)
- **prob**: Probabilistically sample the translations based on the square-root of their frequencies ("root-weighted" in the paper)
- **first**: Deterministically choose the first translation in the list
- **max**: Deterministically choose the translation with maximum frequency in the monolingual data.

### Preprocessed Data
After the pseudo translations, we pre-process that data into PyTorch tensors, to reduce the pre-processing burden on TPU for final training. This maps the last token for every corresponding word in the monolingual data and the pseudo translated data for alignment later. The output is stored in a json format. For this, run the following command:
```shell
python3 preprocess_token_mappings.py\
    --mono path_to_monolingual_data\
    --translated path_to_pseudo_translated_data\
    --vocab_file path_to_vocab_file\
    --max_length 128\
    --outfile path_to_output_preprocessed_json_file
```
`--mono` : Path to the monolingual (text) data<br>
`--translated` : Path to pseudo translated (text) data (created above)<br>
`--vocab_file` : Path to WordPiece vocabulary file used in the pre-trained MLM model. A sample vocab file is present in the [vocab_files](https://github.com/yashkhem1/RelateLM/blob/master/vocab_files) folder.<br>
`--max_length` : Maximum sequence length (Taken as 128 in our experiments)<br>
`--outfile` : Path to output json file for preprocessed data

### Training
The final step is the actual training using Alignment loss. For this we first set the Google Cloud TPU IP address in the console environment using:
```shell
export TPU_IP_ADDRESS=10.89.50.170	#Enter the private IP address of your TPU here
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```
After that, we can start training by running:
```shell
python3 train_alignment_loss.py\
    --files preprocessed_data_l1_l2 preprocessed_data_l2_l1\
    --load path_to_input_model_checkpoint\
    --[is_tf]\
    --ckpt path_to_output_model_checkpoint\
    --bert_config path_to_bert_config\
    --[loss_type mse/cstv]
```
`--files` : Path to the preprocessed data (created above), which can be for both direction for a pair of languages <br>
`--load` : Path to the model checkpoint which is pretrained on MLM. Note that, in case of Tensorflow, the model checkpoints are saved as *XY*.index, *XY*.meta  *XY*.data... etc. So for this case, use *XY* <br>
`--is_tf` : Used if the input model checkpoint is a Tensorflow checkopoint<br>
`--ckpt` : Path to output PyTorch HuggingFace model<br>
`--bert_config` : Path to BERT config JSON file which is required by HuggingFace. A sample config file is present in the [configs](https://github.com/yashkhem1/RelateLM/blob/master/configs) folder.<br>
`--loss_type` : **mse** or **cstv** for MSE and Contrastive loss respectively. By default, the loss is MSE.

## Fine-Tuning on Downstream Tasks
We fine tune of 3 different tasks. The dataset procurement, data cleaning and fine-tuning steps are as follows:

1. Named Entity Recognition :<br>
The dataset is obtained from XTREME Dataset(for en and hi) and WikiAnn NER (for pa, gu, bn, or, as). For preprocessing the WikiAnn NER dataset files, use  "Fine Tuning/Utility Files/wikiann_preprocessor.py" as follows:
```shell
python3 wikiann_preprocessor.py --infile language/language-train.txt --outfile language/train-language.tsv
```
Use the "Fine Tuning/NER_Fine_Tuning.ipynb" for NER evaluation.<br><br> 

POS Tagging and Doc Classification : The datasets for POS Tagging and Doc Classification has been obtained from (Indian Language Technology Proliferation and Deployment Centre)[http://tdil-dc.in/]. Exact links for datasets are available in "Fine Tuning/Utility Files/tdil_dataset.txt".<br><br>
2. Part of Speech Tagging :<br>
Preprocess the data using the preprocessing files from "Fine Tuning/Utility Files/POS/". The "file to language mapping" has been included in "Fine Tuning/Utility Files/POS/Language to File Mapping.txt". Then combines the files using "Fine Tuning/Utility Files/POS/files_combiner.py" to create the train-test splits.
```shell
python3 pos_preprocessor.py --input_folder Language_Raw_Files/ --output_folder Language_POS_Data/
python3 files_combiner.py   --input_folder Language_POS/ --output_folder datasets/ --l_code_actual language_code_as_per_ISO_639 --l_code_in_raw_data language_code_as_per_tdil_dataset
```
We use the (BIS Tagset)[https://www.aclweb.org/anthology/W12-5012.pdf] as the POS tags. The Indian Languages are already tagged with the BIS Tagset whereas the English Dataset is labelled with Penn Tagset. To convert the Penn to BIS, use "Fine Tuning/Utility Files/convert_penn_to_bis.py" to run the following command on the directory containing preprocessed POS dataset files:
```shell
python3 convert_penn_to_bis.py --input_folder English_POS_Penn/ --output_folder English_POS_BIS/
```
Use the "Fine Tuning/POS_Fine_Tuning.ipynb" for POS evaluation.<br> 

<br>
3. Document Classification

## Miscellaneous
### transliterate_monolingual.py
Used for transliterating monolingual data to another languages's script. To use, run:
```shell
python3 transliterate_monolingual.py\
    --mono path_to_monolingual_data\
    --outfile path_to_output_transliterated_data\
    --l1 source_lang\
    --l2 target_lang
```
`--mono` : Path to the monolingual (text) data<br>
`--outfile` : Path to output transliterated (text) file<br>
`--l1` : Code for source language
`--l2` : Code for target language

### statistics.py
Contains various other statistics such as pseudo_translation statistics (*get_pseudo_translation_statistics*), common words among the frequency files (*compare_freq_freq*) etc.

### BLEU.py
Used to calculate BLEU score between Ground truth data and Translated data. To use, run
```shell
python3 BLEU.py --mono path_to_ground_truth_data\
    --pseudo path_to_translated_file
```
