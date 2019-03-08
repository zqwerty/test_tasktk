import tensorflow as tf
import Config

class e2e_user():
    def __init__(self, goal_vocab_size, usr_vocab_size, sys_vocab_size, start_token, end_token, Train=True):
        config = Config.config
        self.goal_vocab_size = goal_vocab_size
        self.usr_vocab_size = usr_vocab_size
        self.sys_vocab_size = sys_vocab_size
        self.goal_embedding_size = config.goal_embedding_size
        self.usr_embedding_size = config.usr_embedding_size
        self.sys_embedding_size = config.sys_embedding_size
        self.dropout = config.dropout
        self.layer_num = config.layer_num
        self.hidden_state_num = config.hidden_state_num
        self.start_token = start_token
        self.end_token = end_token
        self.ifTrain = Train
        self.sys_max_len = 15
        self.usr_max_len = 20

        self.graph = tf.Graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=sess_config)

        self._build_graph()



    # Build graph
    def _build_graph(self):
        
        self.posts = tf.placeholder(tf.int32, (None, None, self.sys_max_len), 'enc_inps')  # batch*sen*max_len
        self.posts_length = tf.placeholder(tf.int32, (None, None), 'enc_lens')  # batch*sen
        self.goals = tf.placeholder(tf.int32, (None, None), 'goal_inps')  # batch*len
        self.goals_length = tf.placeholder(tf.int32, (None,), 'goal_lens')  # batch
        self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
        self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch
        
        # initialize the training process
        self.learning_rate = tf.Variable(float(self.lr), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.lr_decay)
        self.global_step = tf.Variable(0, trainable=False)
        
        _, init_encoder_state = self._goal_encode()
        encoder_states, _ = self._system_encode()
        _, context = self._history_encode(init_encoder_state, encoder_states)
        self._decode(context)
        
        # calculate the gradient of parameters and update
        self.params = tf.trainable_variables()
        opt = tf.train.AdamOptimizer(self.learning_rate)
        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                                          global_step=self.global_step)

		# save checkpoint
        self.latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=5,
                                           pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=1, 
                                         pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def _goal_encode(self, embed=None):
        # build the embedding table and embedding input
        if embed is None:
            # initialize the embedding randomly
            self.goal_embed = tf.get_variable('goal_embed', [self.goal_vocab_size, self.goal_embedding_size], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.goal_embed = tf.get_variable('goal_embed', dtype=tf.float32, initializer=embed)
        encoder_input = tf.nn.embedding_lookup(self.goal_embed, self.goals) #batch*len*unit
        
        cell_enc = tf.nn.rnn_cell.GRUCell(self.hidden_state_num)
        with tf.variable_scope('goal_encoder'):
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_enc, encoder_input,
                                                              self.goals_length, dtype=tf.float32, scope="goal_rnn")
        return encoder_output, encoder_state

    def _system_encode(self, embed=None):
        # build the embedding table and embedding input
        if embed is None:
            # initialize the embedding randomly
            self.sys_embed = tf.get_variable('sys_embed', [self.sys_vocab_size, self.sys_embedding_size], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.sys_embed = tf.get_variable('sys_embed', dtype=tf.float32, initializer=embed)
        batch_size, sentence_max_len = tf.shape(self.posts)[0], tf.shape(self.posts)[1]
        posts_input = tf.reshape(self.posts, [batch_size, -1])
        encoder_input = tf.nn.embedding_lookup(self.sys_embed, posts_input) #batch*(sen*max_len)*unit
        
        cell_enc = tf.nn.rnn_cell.GRUCell(self.hidden_state_num)
        with tf.variable_scope('system_encoder'):
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_enc, encoder_input, dtype=tf.float32, scope="sys_rnn")
        
        w = tf.range(sentence_max_len) * self.sys_max_len
        W = tf.tile(tf.expand_dims(w, 0), [batch_size, 1])
        ends = W + self.posts_length
        rows = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, sentence_max_len])
        indices = tf.concat(tf.expand_dims(rows,-1), tf.expand_dims(ends,-1), 2)
        outputs = tf.gather_nd(encoder_output, indices) #batch*sen*unit
        return outputs, encoder_state

    def _history_encode(self, init_state, inputs):
        cell_enc = tf.nn.rnn_cell.GRUCell(self.hidden_state_num)
        sentence_len = tf.count_nonzero(self.posts_length, 1)
        with tf.variable_scope('history_encoder'):
            outputs, state = tf.nn.dynamic_rnn(cell_enc, inputs, sentence_len,
                                               init_state=init_state, dtype=tf.float32, scope="history_rnn")
        return outputs, state

    def _decode(self, dec_start):
        # deal with original data to adapt encoder and decoder
        batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
        self.responses_length = self.origin_responses_length - 1
        self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0] # no eos_id
        self.responses_target = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1] # no go_id
        decoder_len = decoder_len - 1
        self.decoder_mask = tf.sequence_mask(self.responses_length, decoder_len)
        
        # get output projection function
        output_fn = tf.layers.Dense(self.usr_vocab_size)
        
        # construct cell and helper
        cell_dec = tf.nn.rnn_cell.GRUCell(self.hidden_state_num)
        train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, self.responses_length)
        infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, 
                                                                tf.fill([batch_size], self.start_token), self.end_token)
        
        # build decoder (train)
        with tf.variable_scope('decoder'):
            decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec, train_helper, dec_start, output_layer=output_fn)
            train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished=True, scope="decoder_rnn")
            self.decoder_output = train_outputs.rnn_output
            self.decoder_loss = tf.contrib.seq2seq.sequence_loss(self.decoder_output, self.responses_target, 
                                                                 self.decoder_mask)
        
        # build decoder (test)
        with tf.variable_scope('decoder', reuse=True):
            decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell_dec, infer_helper, dec_start, output_layer=output_fn)
            infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished=True,
                                                                    maximum_iterations=self.usr_max_len, 
                                                                    scope="decoder_rnn")
            self.decoder_distribution = infer_outputs.rnn_output
            self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
                                                       [2, self.usr_vocab_size-2], 2)[1], 2) + 2 #for removing PAD=0, UNK=1

    def step(self, session, data, forward_only=False):
        input_feed = {self.posts: None, # feed with data (temp)
                      self.posts_length: None,
                      self.goals: None,
                      self.goals_length: None,
                      self.origin_responses: None,
                      self.origin_responses_length: None}
        if forward_only:
            output_feed = [self.decoder_loss, self.generation_index] # perplexity=tf.exp(self.decoder_loss)
        else:
            output_feed = [self.decoder_loss, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed) 

    def store_checkpoint(self, sess, path, key):
        if key == "latest":
            self.latest_saver.save(sess, path, global_step = self.global_step)
        else:
            self.best_saver.save(sess, path, global_step = self.global_step)       
    
    # Train step
    def train(self, batch_input):
        with self.sess.as_default():
            with self.graph.as_default():
                loss, grad, _ = self.step(self.sess, data, forward_only=False)
    
    # Evalution step, output result
    def evaluate(self, batch_input):
        pass
