#Transilterate Gujarati monolingual data to Hindi (only if not done before)
python3 transliterate_monolingual.py --mono Wikipedia/Gujarati.txt --outfile Wikipedia/Gujarati_t13n_Hindi.txt --l1 guj --l2 hin

#Frequency File
python3 statistics.py --stats word_freq --mono Wikipedia/Gujarati_t13n_Hindi.txt --outfile Statistics/Gujarati_t13n_Hindi_freq.pkl
python3 statistics.py --stats word_freq --mono Wikipedia/Hindi.txt --outfile Statistics/Hindi_freq.pkl

#Bilingual Lexicon
python3 create_bilingual_dictionary.py --l1 hin --l2 guj --bilingual Bilingual_Lexicons/Hindi-Gujarati Bilingual_Lexicons/Gujarati-Hindi --o1 Bilingual_Lexicons/Hindi_t13n_Gujarati_dict.pkl --o2 Bilingual_Lexicons/t13n_Gujarati_Hindi_dict.pkl --freq_1 Statistics/Hindi_freq.pkl --freq_2 Statistics/Gujarati_t13n_Hindi_freq.pkl --transliterate --trans_lang l1

#Pseudo Translation
python3 create_pseudo_translation.py --mono Wikipedia/Hindi.txt --dict_path Bilingual_Lexicons/Hindi_t13n_Gujarati_dict.pkl --outfile Wikipedia/guj_t13n_hin_pseudo_hin_weighted.txt --replace problin
python3 create_pseudo_translation.py --mono Wikipedia/Gujarati_t13n_Hindi.txt --dict_path Bilingual_Lexicons/t13n_Gujarati_Hindi_dict.pkl --outfile Wikipedia/hin_pseudo_guj_t13n_hin_weighted.txt --replace problin

#Preprocessed data
python3 preprocess_token_mappings.py --mono Wikipedia/Hindi.txt --t13nlated Wikipedia/guj_t13n_hin_pseudo_hin_weighted.txt --vocab_file vocab_files/hindi_gujarati_t13n_hindi_vocab.txt --max_length 128 --outfile preprocessed_data/hin_guj_t13n_hin_weighted.json
python3 preprocess_token_mappings.py --mono Wikipedia/Gujarati_t13n_Hindi.txt --t13nlated Wikipedia/hin_pseudo_guj_t13n_hin_weighted.txt --vocab_file vocab_files/hindi_gujarati_t13n_hindi_vocab.txt --max_length 128 --outfile preprocessed_data/guj_t13n_hin_hin_weighted.json

#Final Training
export TPU_IP_ADDRESS=10.89.50.170	#Enter the private IP address of your TPU here
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
python3 train_alignment_loss.py --files preprocessed_data/hin_guj_t13n_hin_weighted.json preprocessed_data/guj_t13n_hin_hin_weighted.json --load tf_ckpt/mbert_Gujarati_t13n_hindi_bert/model.ckpt-24000 --is_tf --ckpt ckpt/hin_guj_t13n_hin_weighted_aligned_bert_24000.pt --bert_config configs/config.json