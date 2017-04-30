import os
import pickle
import copy
import numpy as np

CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }

def text_to_ids(source_text, target_text, source_vocab_to_int, \
    target_vocab_to_int):
    src_i_t = [[source_vocab_to_int[_] for _ in sent.split()] \
        for sent in source_text.split('\n')]
    tgt_i_t = [[target_vocab_to_int[_] for _ in sent.split()] \
        for sent in target_text.split('\n')]

    for row in tgt_i_t:
        row = row.append(target_vocab_to_int['<EOS>'])

    return (src_i_t, tgt_i_t)

def load_data(path):
    ### Load Dataset from File
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def preprocess(source_text, target_text):
    # Preprocess
    source_text = source_text.lower()
    target_text = target_text.lower()

    source_vocab_to_int, source_int_to_vocab = create_lookup_tables(source_text)
    target_vocab_to_int, target_int_to_vocab = create_lookup_tables(target_text)

    source_text, target_text = text_to_ids(source_text, target_text, \
        source_vocab_to_int, target_vocab_to_int)

    return (source_text, target_text), \
        (source_vocab_to_int, target_vocab_to_int), \
        (source_int_to_vocab, target_int_to_vocab)


def create_lookup_tables(text):
    ### Create lookup tables for vocabulary
    vocab = set(text.split())
    vocab_to_int = copy.copy(CODES)

    for v_i, v in enumerate(vocab, len(CODES)):
        vocab_to_int[v] = v_i

    int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab

def save_params(params):
    ### Save parameters to file
    pickle.dump(params, open('params.p', 'wb'))

def load_params():
    ### Load parameters from file
    return pickle.load(open('params.p', mode='rb'))

def batch_data(source, target, batch_size):
    ### Batch source and target together
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]
        yield np.array(pad_sentence_batch(source_batch)), \
            np.array(pad_sentence_batch(target_batch))

def pad_sentence_batch(sentence_batch):
    ### Pad sentence with <PAD> id
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [CODES['<PAD>']] * (max_sentence - len(sentence))
            for sentence in sentence_batch]
