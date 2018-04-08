# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense

import numpy as np
import os
import data_utils


class attention_seq2seq():
    def __init__(self):
        self.embedding_size = 128
        self.vocab_size = 1000
        self.hidden_size = 128
        self.num_layers = 1
        self.learning_rate = 0.01
        self.batch_size = 128

    def build_model(self):
        self.init_placeholders()
        self.build_encoder()
        self.build_decoder()

        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decorder_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None], name='targets')

        self.source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        self.target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length, name='max_target_len')

    def build_encoder(self):
        """
        encoder
        :return:
        """
        print('build encoder...')
        encoder_embed_input = tf.contrib.layers.embed_sequence(ids=self.inputs, vocab_size=self.vocab_size,
                                                               embed_dim=self.embedding_size)

        def lstm_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_layers)])
        self.encoder_output, self.encoder_state = tf.nn.dynamic_rnn(cell=cells,
                                                                    inputs=encoder_embed_input,
                                                                    sequence_length=self.source_sequence_length,
                                                                    dtype=tf.float32)

    def build_decoder(self):
        """
        decoder
        :return:
        """
        print('build decoder with attention...')
        decoder_embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, self.decoder_input)

        # 2.1 add attention
        def build_decoder_cell_attention():
            attention_states = self.encoder_outputs
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                num_units=self.hidden_size,
                memory=attention_states,
                memory_sequence_length=self.source_sequence_length)
            decoder_cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.hidden_size)

            return decoder_cell

        decoder_cells = tf.contrib.rnn.MultiRNNCell([build_decoder_cell_attention() for _ in range(self.num_layers)])
        # 全连接
        output_layer = Dense(self.vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 4. Training decoder
        with tf.variable_scope("decode"):
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.target_sequence_length,
                                                                time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cells,
                                                               helper=training_helper,
                                                               initial_state=self.encoder_states,
                                                               output_layer=output_layer)
            self.training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                                   impute_finished=True,
                                                                                   maximum_iterations=self.max_target_sequence_length)
        # 5. Predicting decoder
        # 与training共享参数
        with tf.variable_scope("decode", reuse=True):
            '''
            tf.constant, 创建一个常量tensor
            tf.tile(input, multiples, name=None) 复制
            '''
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([data_utils.start_token], dtype=tf.int32), [self.batch_size],
                                   name='start_tokens')
            end_tokens = data_utils.end_tokens
            # use greedy in predict phrase
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_embeddings,
                                                                         start_tokens=start_tokens,
                                                                         end_token=end_tokens)
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cells,
                                                                 helper=predicting_helper,
                                                                 initial_state=self.encoder_states
                                                                 , output_layer=output_layer)
            self.predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                                     impute_finished=True,
                                                                                     maximum_iterations=self.max_target_sequence_length)

    def loss(self):
        training_logits = tf.identity(self.training_decoder_output.rnn_output, name='logits')
        predicting_logits = tf.identity(self.predicting_decoder_output.sample_id, name='predictions')
        masks = tf.sequence_mask(lengths=self.target_sequence_length,
                                 maxlen=self.max_target_sequence_length, dtype=tf.float32, name='masks')
        with tf.name_scope("optimization"):
            self.cost = tf.contrib.seq2seq.sequence_loss(logits=training_logits, targets=self.targets, weights=masks)
            tf.summary.scalar('loss', self.loss)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = self.optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = self.optimizer.apply_gradients(capped_gradients)

    def train(self, sess, encoder_inputs, encoder_inputs_length,
              decoder_inputs, decoder_inputs_length):
        outputs = sess.run([self.loss, self.summary_op], feed_dict={self.inputs: encoder_inputs,
                                                                    self.decoder_input: decoder_inputs,
                                                                    self.targets: decoder_inputs,
                                                                    self.source_sequence_length: encoder_inputs_length,
                                                                    self.target_sequence_length: decoder_inputs_length})

        return outputs[0], outputs[1]


def train():
    train_set = []

    epochs = 10
    with tf.Session() as sess:
        model = attention_seq2seq()
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            for source_seq, trarget_seq in train_set:
                encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length = prepare_train_batch(
                    source_seq, trarget_seq)
                model.train(sess=sess,
                            encoder_inputs=encoder_inputs,
                            encoder_inputs_length=encoder_inputs_length,
                            decoder_inputs=decoder_inputs,
                            decoder_inputs_length=decoder_inputs_length)


if __name__ == '__main__':
    train()