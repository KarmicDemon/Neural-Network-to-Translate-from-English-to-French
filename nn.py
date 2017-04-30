import helper as h
import numpy as np
import tensorflow as tf

class NeuralTranslationEN_FR(object):
    def __init__(self):
        self.source_lang = h.load_data('data/small_vocab_en')
        self.target_lang = h.load_data('data/small_vocab_fr')

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

            train_output = self.decoding_layer_train(encoder_state, rnn,
                dec_embed_input, sequence_length, scope, output_fn, keep_prob)

            scope.reuse_variables()

            infer_output = self.decoding_layer_infer(encoder_state, rnn,
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

    def get_accuracy(self, target, logits):
        max_seq = max(target.shape[1], logits.shape[1])

        if max_seq - target.shape[1]:
            target = np.pad(target, [(0, 0), (0, max_seq - target.shape[1]), \
                (0, 0)], 'constant')

        if max_seq - logits.shape[1]:
            logits = np.pad(logits, [(0, 0), (0,max_seq - logits.shape[1]), \
                (0,0)], 'constant')

        return np.mean(np.equal(target, np.argmax(logits, 2)))

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
        layer = self.encoding_layer(layer, rnn_size, num_layers, keep_prob)

        dec_input = self.process_decoding_input(target_data, target_vocab_to_int,
            batch_size)
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,
            dec_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

        train, infer = self.decoding_layer(dec_embed_input, dec_embeddings,
            layer, target_vocab_size, sequence_length, rnn_size, num_layers,
            target_vocab_to_int, keep_prob)

        return train, infer

    def run(self, epochs = 3, batch_size = 512, rnn_size = 128, num_layers = 1,
        encoding_embedding_size = 200, decoding_embedding_size = 200,
        learning_rate = 0.01, k_p = .75):
        max_target_sent_length = max([len(sent) for sent in \
            self.source_int_text])

        train_graph = tf.Graph()
        with train_graph.as_default():
            input_data, targets, lr, keep_prob = self.model_placeholders()
            sequence_length = tf.placeholder_with_default( \
                max_target_sent_length, None, name = 'sequence_length')
            input_shape = tf.shape(input_data)

            train, infer = self.seq2seq_model(tf.reverse(input_data, [-1]),
                targets, keep_prob, batch_size, sequence_length,
                len(self.source_vocab_to_int), len(self.target_vocab_to_int),
                encoding_embedding_size, decoding_embedding_size, rnn_size,
                num_layers, self.target_vocab_to_int)

            tf.identity(infer, 'logits')

            with tf.name_scope('optimization'):
                cost = tf.contrib.seq2seq.sequence_loss(train, targets,
                    tf.ones([input_shape[0], sequence_length]))
                optimizer = tf.train.AdamOptimizer(learning_rate =
                    learning_rate)
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) \
                    for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

        train_source = self.source_int_text[batch_size:]
        train_target = self.target_int_text[batch_size:]
        valid_source = h.pad_sentence_batch(self.source_int_text[:batch_size])
        valid_target = h.pad_sentence_batch(self.target_int_text[:batch_size])

        with tf.Session(graph = train_graph) as s:
            s.run(tf.global_variables_initializer())

            for e in range(1, epochs + 1):
                for idx, (source_batch, target_batch) in enumerate(
                    h.batch_data(train_source, train_target, batch_size)):

                    _, loss = s.run([train_op, cost], feed_dict = {
                        input_data : source_batch,
                        targets : target_batch,
                        lr : learning_rate,
                        sequence_length : target_batch.shape[1],
                        keep_prob : k_p
                    })

                    batch_train_logits = s.run(infer, feed_dict = {
                        input_data : source_batch,
                        keep_prob : 1.0
                    })

                    batch_valid_logits = s.run(infer, feed_dict = {
                        input_data : valid_source,
                        keep_prob : 1.0
                    })

                    train_acc = self.get_accuracy(target_batch,
                        batch_train_logits)
                    valid_acc = self.get_accuracy(np.array(valid_target),
                        batch_valid_logits)

                    print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy:' \
                        ' {:>6.3f}, Validation Accuracy: {:>6.3f},'\
                        ' Loss: {:>6.3f}'.format(e, idx, \
                        len(self.source_int_text) // batch_size, train_acc, \
                        valid_acc, loss))

        saver = tf.train.Saver()
        saver.save(s, 'checkpoints/dev')
        h.save_params('checkpoints/dev')
        print('Done')


if __name__ == '__main__':
    clf = NeuralTranslationEN_FR()
    clf.run()
