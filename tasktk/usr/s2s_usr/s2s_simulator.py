"""
Created on Feb 13, 2019

@author: xiul
"""

import argparse, json

from tasktk.usr.simulator import User_Simulator
from .lstm_encoder_decoder import lstm_SeqToSeq

decoder_beam_size = 5
decoder_sampling = 0
unk_word = 'unk'


class S2S_Simulator(User_Simulator):
    """An aggregation of user simulator components."""
    def __init__(self, nlu_model, tracker, policy, nlg_model, mode=0):
        """
        Args:
            nlu_model (NLU): An instance of NLU class.
            tracker (Tracker): An instance of Tracker class.
            policy (User_Policy): An instance of Policy class.
            nlg_model (NLG): An instance of NLG class.
            mode (int): 0 for utterance level and 1 for dialog act level.
        """
        self.nlu_model = nlu_model
        self.tracker = tracker
        self.policy = policy
        self.nlg_model = nlg_model
        self.mode = mode
        self.last_state = []
        self.policy.init_session()

    def response(self, input):
        """
        Generate the response of user.
        Args:
            input: Preorder system output, a 1) string if self.mode == 0, else 2) dialog act if self.mode == 1.
        Returns:
            output (str): User response, a 1) string if self.mode == 0, else 2) dialog act if self.mode == 1.
            session_over (boolean): True to terminate session, else session continues.
            reward (float): The reward given by the user.
        """
        if self.mode == 0: # dialog act level
            sys_act = self.nlu_model.parse(input)
            state = self.tracker.update(self.last_state, sys_act)
            action, session_over, reward = self.policy.predict(None, sys_act)
            output = self.nlg_model.generate(action)
        elif self.mode == 1:  # specifically for MDBT testing
            action, session_over, reward = self.policy.predict(None, input)  # see policy_agenda_multiwoz predict
            output = self.nlg_model.generate(action)  # templated nlg
        else:
            raise Exception('Unknown dialog mode: {}'.format(self.mode))
        # self.last_state = state
        
        #print('\tusr: ' + '{}'.format(action))
        #print('usr nl:', output)
        return output, session_over, reward

    def load_model(self, model_path):
        """ load the trained S2S USim model """
        
        model_params = pickle.load(open(model_path, 'rb'))
    
        encoder_hidden_size = model_params['model']['Wah'].shape[0]
        decoder_hidden_size = model_params['model']['Wd'].shape[0]
        decoder_output_size = model_params['model']['Wd'].shape[1]
    
        if model_params['params']['model'] == 'lstm': # lstm
            encoder_input_size = model_params['model']['e_WLSTM'].shape[0] - encoder_hidden_size - 1
            decoder_input_size = model_params['model']['WLSTM'].shape[0] - decoder_hidden_size - 1
            rnnmodel = lstm_SeqToSeq(encoder_input_size, encoder_hidden_size, decoder_input_size, decoder_hidden_size, decoder_output_size)
     
        rnnmodel.model = copy.deepcopy(model_params['model'])
    
        model_params['params']['beam_size'] = decoder_beam_size
        model_params['params']['decoder_sampling'] = decoder_sampling
        
        self.model = rnnmodel
        self.src_word_dict = copy.deepcopy(model_params['src_word_dict'])
        self.tgt_word_dict = copy.deepcopy(model_params['tgt_word_dict'])
        self.inverse_tgt_word_dict = {self.tgt_word_dict[k]:k for k in self.tgt_word_dict.keys()}
        self.params = copy.deepcopy(model_params['params'])
    
    def model_predict(self, input):
        """ model predict on single utterance """
        
        input_len = len(input.lower().split(' '))
        input_ele = {}
        
        input_seq_ix = [0]*input_len
        input_seq_vec = np.zeros((input_len, len(self.src_word_dict)))
        input_seq_tgt_vec = np.zeros((2, len(self.tgt_word_dict)))
        for w_id, w in enumerate(input.lower().split(' ')):
            if w in self.src_word_dict: input_seq_ix[w_id] = self.src_word_dict[w]
            else: input_seq_ix[w_id] = self.src_word_dict[unk_word]
            input_seq_vec[w_id][input_seq_ix[w_id]] = 1.0
        
        input_seq_tgt_vec[0][tgt_word_dict['bos']] = 1.0 
        input_ele['src_seq_ix'] = input_seq_ix
        input_ele['src_seq_rep'] = input_seq_vec
        input_ele['tgt_seq_rep'] = input_seq_tgt_vec
        
        pred_ys, pred_words = self.model.forward(self.inverse_tgt_word_dict, input_ele, self.params, predict_model=True)
        #pred_ys, pred_words = self.model.beam_forward(self.inverse_tgt_word_dict, input_ele, self.params, predict_model=True)
        
        pred_sentence = ' '.join(pred_words[:-1])    
        return pred_sentence
            
    
    def init_session(self):
        """Init the parameters for a new session."""
        self.last_state = []
        self.policy.init_session()