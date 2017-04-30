import helper as h
import numpy as np
import tensorflow as tf

class NeuralTranslationEN_FR(object):
    def __init__(self):
        self.source_lang = helper.load('data/small_vocab_en')
        self.target_lang = helper.load('data/small_vocab_fr')

        ((self.source_int_text, self.target_int_text), \
            (self.source_vocab_to_int, self.target_vocab_to_int), \
            _) = h.preprocess(self.source_lang, self.target_lang)

    def decoding_layer(self, dec_embed_input, dec_embeddings, encoder_state,
                   vocab_size, sequence_length, rnn_size, num_layers,
                   target_vocab_to_int, keep_prob):

        rnn = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        rnn = tf.contrib.rnn.DropoutWrapper(rnn, output_keep_prob = keep_prob)
        rnn = tf.contrib.rnn.MultiRNNCell([rnn] * num_layers)

        with tf.variable_scope("decoding_scope") as scope:
            output_fn = lambda x : tf.contrib.layers.fully_connected(x,
            vocab_size, activation_fn = None, scope = scope)

            train_output = decoding_layer_train(encoder_state, rnn,
                dec_embed_input, sequence_length, scope, output_fn, keep_prob)

            scope.reuse_variables()

            infer_output = decoding_layer_infer(encoder_state, rnn,
                dec_embeddings, target_vocab_to_int['<GO>'],
                target_vocab_to_int['<EOS>'], sequence_length, vocab_size,
                scope, output_fn, keep_prob)

        return train_output, infer_output

    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings,
        start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size,
        decoding_scope, output_fn, keep_prob):

        dfi = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn,
            encoder_state, dec_embeddings, start_of_sequence_id,
            end_of_sequence_id, (maximum_length - 1), vocab_size,
            name = 'simple_decoder_fn_inference')

        outputs, final_state, fcs = tf.contrib.seq2seq.dynamic_rnn_decoder(\
            dec_cell, dfi, scope = decoding_scope,
            name = 'infererence_dynamic_rnn_decoder')

        return outputs

    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input,
        sequence_length, decoding_scope, output_fn, keep_prob):
        decoder = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)

        outputs, final_state, fcs = tf.contrib.seq2seq.dynamic_rnn_decoder(\
            dec_cell,decoder, dec_embed_input, sequence_length,
            scope = decoding_scope, name = 'training_dynamic_rnn_decoder')

        return output_fn(outputs)

    def encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob):
        rnn = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        rnn = tf.contrib.rnn.DropoutWrapper(rnn, output_keep_prob = keep_prob)
        rnn = tf.contrib.rnn.MultiRNNCell([rnn] * num_layers)
        _, initial_state = tf.nn.dynamic_rnn(rnn, rnn_inputs, \
            dtype = tf.float32)

        return initial_state

    def model_placeholders(self):
        _input = tf.placeholder(tf.int32, [None, None], name = 'input')
        _targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
        _lr = tf.placeholder(tf.float32, name = 'learning_rate')
        _kp = tf.placeholder(tf.float32, name = 'keep_prob')

        return _input, _targets, _lr, _kp

    def process_decoding_input(self, target_data, target_vocab_to_int,
        batch_size):
        _ = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        new_data = tf.concat([tf.fill([batch_size, 1], \
            target_vocab_to_int['<GO>']), _], 1)

        return new_data

    def seq2seq_model(self, input_data, target_data, keep_prob, batch_size,
        sequence_length, source_vocab_size, target_vocab_size,
        enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
        target_vocab_to_int):
        layer = tf.contrib.layers.embed_sequence(input_data,
        source_vocab_size, enc_embedding_size)
        layer = encoding_layer(layer, rnn_size, num_layers, keep_prob)

        dec_input = process_decoding_input(target_data, target_vocab_to_int,
            batch_size)
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,
            dec_embedding_size]))
            dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        train, infer = decoding_layer(dec_embed_input, dec_embeddings,
            layer, target_vocab_size, sequence_length, rnn_size, num_layers,
            target_vocab_to_int, keep_prob)

        return train, infer

    def run(self, epochs = 3, batch_size = 512, rnn_size = 128, num_layers = 1,
        encoding_embedding_size = 200, decoding_embedding_size = 200,
        learning_rate = 0.01, keep_prob = .75):
        train_graph = tf.Graph()
        with train_graph.as_default():
            input_data, targets, lr, keep_prob = model_placeholders()
