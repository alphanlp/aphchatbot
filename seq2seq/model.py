# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper

import numpy as np
import os
import data_utils


class attention_seq2seq():
    """
    seq2seq模型
    """

    def __init__(self, vocab_size, mode='train'):
        self.embedding_size = 128
        self.vocab_size = vocab_size
        self.hidden_size = 256
        self.num_layers = 2
        self.learning_rate = 0.001
        self.mode = mode

    def build_model(self):
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        self.merged = tf.summary.merge_all()

    def init_placeholders(self):
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        self.batch_size = tf.shape(self.inputs)[0]

        if self.mode == 'train':
            self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decorder_inputs')
            self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
            self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
            self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')

    def build_encoder(self):
        """
        encoder
        :return:
        """
        print('build encoder...')
        with tf.variable_scope('encoder'):
            # encoder_embed_input = tf.contrib.layers.embed_sequence(ids=self.inputs, vocab_size=self.vocab_size,
            #                                                        embed_dim=self.embedding_size)
            self.encoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(params=self.encoder_embeddings, ids=self.inputs)

            def lstm_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return lstm_cell

            cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_layers)])
            self.encoder_outputs, self.encoder_states = tf.nn.dynamic_rnn(cell=cells,
                                                                          inputs=self.encoder_inputs_embedded,
                                                                          sequence_length=self.source_sequence_length,
                                                                          dtype=tf.float32)

    def build_decoder(self):
        """
        decoder
        :return:
        """
        print('build decoder with attention...')
        with tf.variable_scope('decoder'):
            self.decoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))

            # 2.1 add attention
            def build_decoder_cell():
                decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                       initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
                return decoder_cell

            attention_states = self.encoder_outputs
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.hidden_size,
                memory=attention_states,
                memory_sequence_length=self.source_sequence_length)

            decoder_cells_list = [build_decoder_cell() for _ in range(self.num_layers)]
            decoder_cells_list[-1] = attention_wrapper.AttentionWrapper(
                cell=decoder_cells_list[-1],
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.hidden_size
            )

            self.decoder_cells = tf.contrib.rnn.MultiRNNCell(decoder_cells_list)

            initial_state = [state for state in self.encoder_states]
            initial_state[-1] = decoder_cells_list[-1].zero_state(
                batch_size=self.batch_size, dtype=tf.float32)
            self.decoder_initial_state = tuple(initial_state)

            # 全连接
            self.output_layer = Dense(self.vocab_size,
                                      kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            if self.mode == 'train':
                self.interfer()
            elif self.mode == 'decode':
                self.decode()

    def interfer(self):
        # 4. Training decoder
        decoder_embed_input = tf.nn.embedding_lookup(self.decoder_embeddings, self.decoder_input)
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=self.target_sequence_length,
                                                            time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cells,
                                                           helper=training_helper,
                                                           initial_state=self.decoder_initial_state,
                                                           output_layer=self.output_layer)
        self.training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                               impute_finished=True,
                                                                               maximum_iterations=self.max_target_sequence_length)
        self.optimization()

    def optimization(self):
        training_logits = tf.identity(self.training_decoder_output.rnn_output, name='logits')
        masks = tf.sequence_mask(lengths=self.target_sequence_length,
                                 maxlen=self.max_target_sequence_length, dtype=tf.float32, name='masks')
        with tf.name_scope("optimization"):
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=training_logits, targets=self.targets, weights=masks)
            tf.summary.scalar('loss', self.loss)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            gradients = self.optimizer.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = self.optimizer.apply_gradients(capped_gradients)

    def decode(self):
        start_tokens = tf.tile(tf.constant([data_utils.start_token], dtype=tf.int32), [self.batch_size],
                               name='start_tokens')
        end_tokens = data_utils.end_token
        # use greedy in predict phrase
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.decoder_embeddings,
                                                                     start_tokens=start_tokens,
                                                                     end_token=end_tokens)
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell=self.decoder_cells,
                                                             helper=predicting_helper,
                                                             initial_state=self.decoder_initial_state,
                                                             output_layer=self.output_layer)
        self.predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                 impute_finished=True,
                                                                                 maximum_iterations=20)
        self.predicting_logits = tf.identity(self.predicting_decoder_output.sample_id, name='predictions')

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_targets, decoder_inputs_length):
        decoder_inputs = np.delete(decoder_targets, -1, axis=1)
        decoder_inputs = np.c_[np.zeros(len(decoder_inputs), dtype=np.int32), decoder_inputs]
        outputs = sess.run([self.train_op, self.loss, self.merged], feed_dict={self.inputs: encoder_inputs,
                                                                               self.decoder_input: decoder_inputs,
                                                                               self.targets: decoder_targets,
                                                                               self.source_sequence_length: encoder_inputs_length,
                                                                               self.target_sequence_length: decoder_inputs_length})
        return outputs[0], outputs[1], outputs[2]

    def merge(self, sess, train_summary, step):
        tf.summary.scalar("loss", self.loss)
        merged = sess.run(self.merged)
        train_summary.add_summary(merged, global_step=step)

    def predict(self, sess, encoder_inputs, encoder_inputs_length):
        outputs = sess.run([self.predicting_logits], feed_dict={
            self.inputs: encoder_inputs,
            self.source_sequence_length: encoder_inputs_length
        })
        return outputs


def train():
    input_path = "../resources/inputs"
    output_path = "../resources/outputs"
    vocab_path = "../vocab.pickle"

    checkpoint = "../model/checkpoint/model.ckpt"
    batch_size = 128

    index_to_char, char_to_index, vocab_size = data_utils.load_vocab(vocab_path)
    input_sentence, output_sentence = data_utils.format_corpus(char_to_index, input_path, output_path)

    epochs = 100
    with tf.Session() as sess:
        model = attention_seq2seq(vocab_size)
        model.build_model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(1, epochs + 1):
            train_set = data_utils.train_set(input_sentence, output_sentence, batch_size)
            for source_seq, target_seq in train_set:
                encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length = data_utils.prepare_train_batch(
                    source_seq, target_seq)
                _, loss, _ = model.train(sess=sess,
                                         encoder_inputs=encoder_inputs,
                                         encoder_inputs_length=encoder_inputs_length,
                                         decoder_targets=decoder_inputs,
                                         decoder_inputs_length=decoder_inputs_length)
                print("epoch={}, loss={}".format(epoch, loss))

                saver.save(sess, save_path=checkpoint, global_step=epoch)
                print('Model Trained and Saved')



def predit():
    vocab_path = "../vocab.pickle"

    input_sentence = "不是"
    index_to_char, char_to_index, vocab_size = data_utils.load_vocab(vocab_path)
    form_input = []
    for ch in input_sentence:
        try:
            ch = char_to_index[ch]
            form_input.append(ch)
        except KeyError:
            pass
    encoder_inputs, encoder_inputs_length = data_utils.prepare_predict_batch([form_input])
    checkpoint = "../model/checkpoint/model.ckpt-1"

    with tf.Session() as sess:
        model = attention_seq2seq(vocab_size=vocab_size, mode='decode')
        model.build_model()
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=checkpoint)
        predicted_ids = model.predict(sess=sess,
                                      encoder_inputs=encoder_inputs,
                                      encoder_inputs_length=encoder_inputs_length)
        predicted_ids = predicted_ids[0].tolist()
        predicted_ids = predicted_ids[0]
        print(predicted_ids)
        temp = [index_to_char[i] for i in predicted_ids if i != data_utils.end_token]
        print(temp)
        print("".join(temp))


if __name__ == '__main__':
    train()
    # predit()
