'''
Created on Jun 13, 2016

An LSTM Encoder-Decoder

@author: xiul
'''


from .seq_seq import SeqToSeq
from .utils import *


class lstm_SeqToSeq(SeqToSeq):
    def __init__(self, encoder_input_size, encoder_hidden_size, decoder_input_size, decoder_hidden_size, decoder_output_size):
        self.model = {}
        
        # Encoder: Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['e_WLSTM'] = initWeights(encoder_input_size + encoder_hidden_size + 1, 4*encoder_hidden_size)
        
        # connections from diaact to hidden layer
        self.model['Wah'] = initWeights(encoder_hidden_size, 4*decoder_hidden_size)
        self.model['bah'] = np.zeros((1, 4*decoder_hidden_size))
        
        # Recurrent weights: take x_t, h_{t-1}, and bias unit, and produce the 3 gates and the input to cell signal
        self.model['WLSTM'] = initWeights(decoder_input_size + decoder_hidden_size + 1, 4*decoder_hidden_size)
        # Hidden-Output Connections
        self.model['Wd'] = initWeights(decoder_hidden_size, decoder_output_size)*0.1
        self.model['bd'] = np.zeros((1, decoder_output_size))

        self.update = ['Wah', 'bah', 'WLSTM', 'Wd', 'bd', 'e_WLSTM']
        self.regularize = ['Wah', 'WLSTM', 'Wd', 'e_WLSTM']

        self.step_cache = {}
        
    """ Forward """
    def fwdPass(self, Xs, params, **kwargs):
        predict_mode = kwargs.get('predict_mode', False)
        feed_recurrence = params.get('feed_recurrence', 0)
        
        src_Ds = Xs['src_seq_rep']
        tgt_Ds = Xs['tgt_seq_rep'][0:-1]
        
        #print src_Ds.shape, tgt_Ds.shape
        
        # encoder
        e_WLSTM = self.model['e_WLSTM']
        e_n, e_xd = src_Ds.shape
        e_d = self.model['Wah'].shape[0]
        
        e_Hin = np.zeros((e_n, e_WLSTM.shape[0])) # xt, ht-1, bias
        e_Hout = np.zeros((e_n, e_d))
        e_IFOG = np.zeros((e_n, 4*e_d))
        e_IFOGf = np.zeros((e_n, 4*e_d)) # after nonlinearity
        e_Cellin = np.zeros((e_n, e_d))
        e_Cellout = np.zeros((e_n, e_d))
        
        for t in xrange(e_n):
            prev = np.zeros(e_d) if t==0 else e_Hout[t-1]
            e_Hin[t,0] = 1 # bias
            e_Hin[t, 1:1+e_xd] = src_Ds[t]
            e_Hin[t, 1+e_xd:] = prev
            
            # compute all gate activations. dots:
            e_IFOG[t] = e_Hin[t].dot(e_WLSTM)

            e_IFOGf[t, :3*e_d] = 1/(1+np.exp(-e_IFOG[t, :3*e_d])) # sigmoids; these are three gates
            e_IFOGf[t, 3*e_d:] = np.tanh(e_IFOG[t, 3*e_d:]) # tanh for input value
            
            e_Cellin[t] = e_IFOGf[t, :e_d] * e_IFOGf[t, 3*e_d:]
            if t>0: e_Cellin[t] += e_IFOGf[t, e_d:2*e_d]*e_Cellin[t-1]
            
            e_Cellout[t] = np.tanh(e_Cellin[t])
            e_Hout[t] = e_IFOGf[t, 2*e_d:3*e_d] * e_Cellout[t]
        
        # end of Encoder
        encode_Hout = np.array([e_Hout[-1]])
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = encode_Hout.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        n, xd = tgt_Ds.shape
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((n, d))
        IFOG = np.zeros((n, 4*d))
        IFOGf = np.zeros((n, 4*d)) # after nonlinearity
        Cellin = np.zeros((n, d))
        Cellout = np.zeros((n, d))
    
        for t in xrange(n):
            prev = np.zeros(d) if t==0 else Hout[t-1]
            Hin[t,0] = 1 # bias
            Hin[t, 1:1+xd] = tgt_Ds[t]
            Hin[t, 1+xd:] = prev
            
            # compute all gate activations. dots:
            IFOG[t] = Hin[t].dot(WLSTM)
            
            # add diaact vector here
            if feed_recurrence == 0:
                if t == 0: IFOG[t] += Dsh[0]
            else: IFOG[t] += Dsh[0]

            IFOGf[t, :3*d] = 1/(1+np.exp(-IFOG[t, :3*d])) # sigmoids; these are three gates
            IFOGf[t, 3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh for input value
            
            Cellin[t] = IFOGf[t, :d] * IFOGf[t, 3*d:]
            if t>0: Cellin[t] += IFOGf[t, d:2*d]*Cellin[t-1]
            
            Cellout[t] = np.tanh(Cellin[t])
            Hout[t] = IFOGf[t, 2*d:3*d] * Cellout[t]

        Wd = self.model['Wd']
        bd = self.model['bd']
        Y = Hout.dot(Wd)+bd
            
        cache = {}
        if not predict_mode:
            cache['e_WLSTM'] = e_WLSTM
            cache['e_Hout'] = e_Hout
            cache['e_IFOGf'] = e_IFOGf
            cache['e_IFOG'] = e_IFOG
            cache['e_Cellin'] = e_Cellin
            cache['e_Cellout'] = e_Cellout
            cache['e_Hin'] = e_Hin
            
            cache['WLSTM'] = WLSTM
            cache['Hout'] = Hout
            cache['Wd'] = Wd
            cache['IFOGf'] = IFOGf
            cache['IFOG'] = IFOG
            cache['Cellin'] = Cellin
            cache['Cellout'] = Cellout
            cache['Hin'] = Hin
            
            cache['tgt_Ds'] = tgt_Ds
            cache['src_Ds'] = src_Ds
            
            cache['Dsh'] = Dsh
            cache['Wah'] = Wah
            
            cache['feed_recurrence'] = feed_recurrence
            
        return Y, cache
    
    """ Forward pass on prediction """
    def forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 50)
        feed_recurrence = params.get('feed_recurrence', 0)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        src_Ds = Xs['src_seq_rep']
        tgt_Ds = Xs['tgt_seq_rep'][0:-1]
        
        #print src_Ds.shape, tgt_Ds.shape
        
        # encoder
        e_WLSTM = self.model['e_WLSTM']
        e_n, e_xd = src_Ds.shape
        e_d = self.model['Wah'].shape[0]
        
        e_Hin = np.zeros((e_n, e_WLSTM.shape[0])) # xt, ht-1, bias
        e_Hout = np.zeros((e_n, e_d))
        e_IFOG = np.zeros((e_n, 4*e_d))
        e_IFOGf = np.zeros((e_n, 4*e_d)) # after nonlinearity
        e_Cellin = np.zeros((e_n, e_d))
        e_Cellout = np.zeros((e_n, e_d))
        
        for t in xrange(e_n):
            prev = np.zeros(e_d) if t==0 else e_Hout[t-1]
            e_Hin[t,0] = 1 # bias
            e_Hin[t, 1:1+e_xd] = src_Ds[t]
            e_Hin[t, 1+e_xd:] = prev
            
            # compute all gate activations. dots:
            e_IFOG[t] = e_Hin[t].dot(e_WLSTM)

            e_IFOGf[t, :3*e_d] = 1/(1+np.exp(-e_IFOG[t, :3*e_d])) # sigmoids; these are three gates
            e_IFOGf[t, 3*e_d:] = np.tanh(e_IFOG[t, 3*e_d:]) # tanh for input value
            
            e_Cellin[t] = e_IFOGf[t, :e_d] * e_IFOGf[t, 3*e_d:]
            if t>0: e_Cellin[t] += e_IFOGf[t, e_d:2*e_d]*e_Cellin[t-1]
            
            e_Cellout[t] = np.tanh(e_Cellin[t])
            e_Hout[t] = e_IFOGf[t, 2*e_d:3*e_d] * e_Cellout[t]
        
        # end of Encoder
        encode_Hout = np.array([e_Hout[-1]])
        
        #print encode_Hout
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = encode_Hout.dot(Wah) + bah
        
        # Decoder
        WLSTM = self.model['WLSTM']
        xd = tgt_Ds.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = tgt_Ds[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        pred_y = []
        pred_words = []
        
        Y = Hout[0].dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
        
        #print 't0:', probs, np.nanargmax(Y[0]), probs[0][np.nanargmax(Y[0])]
            
        if decoder_sampling == 0: # argmax
            pred_y_index = np.nanargmax(Y[0])
        else: # sampling
            pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
        pred_y.append(pred_y_index)
        pred_words.append(dict[pred_y_index])
        
        time_stamp = 0
        while True:
            if dict[pred_y_index] == 'eos' or time_stamp >= max_len: break
            
            X = np.zeros(xd)
            X[pred_y_index] = 1
            Hin[0,0] = 1 # bias
            Hin[0,1:1+xd] = X
            Hin[0, 1+xd:] = Hout[0]
            
            IFOG[0] = Hin[0].dot(WLSTM)
            if feed_recurrence == 1: IFOG[0] += Dsh[0]
        
            IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
            IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
            C = IFOGf[0, :d]*IFOGf[0, 3*d:]
            Cellin[0] = C + IFOGf[0, d:2*d]*Cellin[0]
            Cellout[0] = np.tanh(Cellin[0])
            Hout[0] = IFOGf[0, 2*d:3*d]*Cellout[0]
            
            Y = Hout[0].dot(Wd) + bd
            maxes = np.amax(Y, axis=1, keepdims=True)
            e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
            probs = e/np.sum(e, axis=1, keepdims=True)
            
            #print 't', time_stamp, probs, np.nanargmax(Y[0])
            
            if decoder_sampling == 0:
                pred_y_index = np.nanargmax(Y[0])
            else:
                pred_y_index = np.random.choice(Y.shape[1], 1, p=probs[0])[0]
            pred_y.append(pred_y_index)
            pred_words.append(dict[pred_y_index])
            
            time_stamp += 1
        
        return pred_y, pred_words
    
    """ Forward pass on prediction with Beam Search """
    def beam_forward(self, dict, Xs, params, **kwargs):
        max_len = params.get('max_len', 50)
        feed_recurrence = params.get('feed_recurrence', 0)
        beam_size = params.get('beam_size', 10)
        decoder_sampling = params.get('decoder_sampling', 0)
        
        src_Ds = Xs['src_seq_rep']
        tgt_Ds = Xs['tgt_seq_rep'][0:-1]
        
        # encoder
        e_WLSTM = self.model['e_WLSTM']
        e_n, e_xd = src_Ds.shape
        e_d = self.model['Wah'].shape[0]
        
        e_Hin = np.zeros((e_n, e_WLSTM.shape[0])) # xt, ht-1, bias
        e_Hout = np.zeros((e_n, e_d))
        e_IFOG = np.zeros((e_n, 4*e_d))
        e_IFOGf = np.zeros((e_n, 4*e_d)) # after nonlinearity
        e_Cellin = np.zeros((e_n, e_d))
        e_Cellout = np.zeros((e_n, e_d))
        
        for t in xrange(e_n):
            prev = np.zeros(e_d) if t==0 else e_Hout[t-1]
            e_Hin[t,0] = 1 # bias
            e_Hin[t, 1:1+e_xd] = src_Ds[t]
            e_Hin[t, 1+e_xd:] = prev
            
            # compute all gate activations. dots:
            e_IFOG[t] = e_Hin[t].dot(e_WLSTM)

            e_IFOGf[t, :3*e_d] = 1/(1+np.exp(-e_IFOG[t, :3*e_d])) # sigmoids; these are three gates
            e_IFOGf[t, 3*e_d:] = np.tanh(e_IFOG[t, 3*e_d:]) # tanh for input value
            
            e_Cellin[t] = e_IFOGf[t, :e_d] * e_IFOGf[t, 3*e_d:]
            if t>0: e_Cellin[t] += e_IFOGf[t, e_d:2*e_d]*e_Cellin[t-1]
            
            e_Cellout[t] = np.tanh(e_Cellin[t])
            e_Hout[t] = e_IFOGf[t, 2*e_d:3*e_d] * e_Cellout[t]
        
        # end of Encoder
        encode_Hout = np.array([e_Hout[-1]])
        
        # diaact input layer to hidden layer
        Wah = self.model['Wah']
        bah = self.model['bah']
        Dsh = encode_Hout.dot(Wah) + bah
        
        WLSTM = self.model['WLSTM']
        xd = tgt_Ds.shape[1]
        
        d = self.model['Wd'].shape[0] # size of hidden layer
        Hin = np.zeros((1, WLSTM.shape[0])) # xt, ht-1, bias
        Hout = np.zeros((1, d))
        IFOG = np.zeros((1, 4*d))
        IFOGf = np.zeros((1, 4*d)) # after nonlinearity
        Cellin = np.zeros((1, d))
        Cellout = np.zeros((1, d))
        
        Wd = self.model['Wd']
        bd = self.model['bd']
        
        Hin[0,0] = 1 # bias
        Hin[0,1:1+xd] = tgt_Ds[0]
        
        IFOG[0] = Hin[0].dot(WLSTM)
        IFOG[0] += Dsh[0]
        
        IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
        IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
        Cellin[0] = IFOGf[0, :d] * IFOGf[0, 3*d:]
        Cellout[0] = np.tanh(Cellin[0])
        Hout[0] = IFOGf[0, 2*d:3*d] * Cellout[0]
        
        # keep a beam here
        beams = [] 
        
        Y = Hout[0].dot(Wd) + bd
        maxes = np.amax(Y, axis=1, keepdims=True)
        e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
        probs = e/np.sum(e, axis=1, keepdims=True)
        
        # add beam search here
        if decoder_sampling == 0: # no sampling
            beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        else:
            beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
        #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
        for ele in beam_candidate_t:
            beams.append((np.log(probs[0][ele]), [ele], [dict[ele]], Hout[0], Cellin[0]))
        
        time_stamp = 0
        while True:
            beam_candidates = []
            for b in beams:
                log_prob = b[0]
                pred_y_index = b[1][-1]
                cell_in = b[4]
                hout_prev = b[3]
                
                if b[2][-1] == "eos": # this beam predicted end token. Keep in the candidates but don't expand it out any more
                    beam_candidates.append(b)
                    continue
        
                X = np.zeros(xd)
                X[pred_y_index] = 1
                Hin[0,0] = 1 # bias
                Hin[0,1:1+xd] = X
                Hin[0, 1+xd:] = hout_prev
                
                IFOG[0] = Hin[0].dot(WLSTM)
                if feed_recurrence == 1: IFOG[0] += Dsh[0]
        
                IFOGf[0, :3*d] = 1/(1+np.exp(-IFOG[0, :3*d])) # sigmoids; these are three gates
                IFOGf[0, 3*d:] = np.tanh(IFOG[0, 3*d:]) # tanh for input value
            
                C = IFOGf[0, :d]*IFOGf[0, 3*d:]
                cell_in = C + IFOGf[0, d:2*d]*cell_in
                cell_out = np.tanh(cell_in)
                hout_prev = IFOGf[0, 2*d:3*d]*cell_out
                
                Y = hout_prev.dot(Wd) + bd
                maxes = np.amax(Y, axis=1, keepdims=True)
                e = np.exp(Y - maxes) # for numerical stability shift into good numerical range
                probs = e/np.sum(e, axis=1, keepdims=True)
                
                if decoder_sampling == 0: # no sampling
                    beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                else:
                    beam_candidate_t = np.random.choice(Y.shape[1], beam_size, p=probs[0])
                #beam_candidate_t = (-probs[0]).argsort()[:beam_size]
                for ele in beam_candidate_t:
                    beam_candidates.append((log_prob+np.log(probs[0][ele]), np.append(b[1], ele), np.append(b[2], dict[ele]), hout_prev, cell_in))
            
            beam_candidates.sort(key=lambda x:x[0], reverse=True)
            #beam_candidates.sort(reverse = True) # decreasing order
            beams = beam_candidates[:beam_size]
            time_stamp += 1

            if time_stamp >= max_len: break
        
        return_candidate = beams[0]
        return return_candidate[1], return_candidate[2]
        #return beams[0][1], beams[0][2]
    
    """ Backward Pass """
    def bwdPass(self, dY, cache):
        Wd = cache['Wd']
        Hout = cache['Hout']
        IFOG = cache['IFOG']
        IFOGf = cache['IFOGf']
        Cellin = cache['Cellin']
        Cellout = cache['Cellout']
        Hin = cache['Hin']
        WLSTM = cache['WLSTM']
        
        tgt_Ds = cache['tgt_Ds']
        src_Ds = cache['src_Ds']
        
        Dsh = cache['Dsh']
        Wah = cache['Wah']
        
        e_Hout = cache['e_Hout']
        e_IFOG = cache['e_IFOG']
        e_IFOGf = cache['e_IFOGf']
        e_Cellin = cache['e_Cellin']
        e_Cellout = cache['e_Cellout']
        e_Hin = cache['e_Hin']
        e_WLSTM = cache['e_WLSTM']
        
        feed_recurrence = cache['feed_recurrence']
        
        # backprop on Decoder      
        n,d = Hout.shape

        # backprop the hidden-output layer
        dWd = Hout.transpose().dot(dY)
        dbd = np.sum(dY, axis=0, keepdims = True)
        dHout = dY.dot(Wd.transpose())

        # backprop the LSTM
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dCellin = np.zeros(Cellin.shape)
        dCellout = np.zeros(Cellout.shape)
        
        dDsh = np.zeros(Dsh.shape)
        
        for t in reversed(xrange(n)):
            dIFOGf[t,2*d:3*d] = Cellout[t] * dHout[t]
            dCellout[t] = IFOGf[t,2*d:3*d] * dHout[t]
            
            dCellin[t] += (1-Cellout[t]**2) * dCellout[t]
            
            if t>0:
                dIFOGf[t, d:2*d] = Cellin[t-1] * dCellin[t]
                dCellin[t-1] += IFOGf[t,d:2*d] * dCellin[t]
            
            dIFOGf[t, :d] = IFOGf[t,3*d:] * dCellin[t]
            dIFOGf[t,3*d:] = IFOGf[t, :d] * dCellin[t]
            
            # backprop activation functions
            dIFOG[t, 3*d:] = (1-IFOGf[t, 3*d:]**2) * dIFOGf[t, 3*d:]
            y = IFOGf[t, :3*d]
            dIFOG[t, :3*d] = (y*(1-y)) * dIFOGf[t, :3*d]
            
            # backprop matrix multiply
            dWLSTM += np.outer(Hin[t], dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
      
            if t > 0: dHout[t-1] += dHin[t,1+tgt_Ds.shape[1]:]
            
            if feed_recurrence == 0:
                if t == 0: dDsh[t] = dIFOG[t]
            else: 
                dDsh[0] += dIFOG[t]
        
        encode_Hout = np.array([e_Hout[-1]])
        
        # backprop to the diaact-hidden connections
        dWah = encode_Hout.transpose().dot(dDsh)
        dbah = np.sum(dDsh, axis=0, keepdims = True)
        
        # backprop on Encoder
        de_Hout = np.zeros(e_Hout.shape)
        de_Hout[-1] = dDsh.dot(Wah.transpose())[0]
        
        de_IFOG = np.zeros(e_IFOG.shape)
        de_IFOGf = np.zeros(e_IFOGf.shape)
        de_WLSTM = np.zeros(e_WLSTM.shape)
        de_Hin = np.zeros(e_Hin.shape)
        de_Cellin = np.zeros(e_Cellin.shape)
        de_Cellout = np.zeros(e_Cellout.shape)
        
        e_n,e_d = e_Hout.shape
        for t in reversed(xrange(e_n)):
            de_IFOGf[t, 2*e_d:3*e_d] = e_Cellout[t] * de_Hout[t]
            de_Cellout[t] = e_IFOGf[t, 2*e_d:3*e_d] * de_Hout[t]
            
            de_Cellin[t] += (1-e_Cellout[t]**2) * de_Cellout[t]
            
            if t>0:
                de_IFOGf[t, e_d:2*e_d] = e_Cellin[t-1] * de_Cellin[t]
                de_Cellin[t-1] += e_IFOGf[t, e_d:2*e_d] * de_Cellin[t]
            
            de_IFOGf[t, :e_d] = e_IFOGf[t,3*e_d:] * de_Cellin[t]
            de_IFOGf[t,3*e_d:] = e_IFOGf[t, :e_d] * de_Cellin[t]
            
            # backprop activation functions
            de_IFOG[t, 3*e_d:] = (1-e_IFOGf[t, 3*e_d:]**2) * de_IFOGf[t, 3*e_d:]
            y = e_IFOGf[t, :3*e_d]
            de_IFOG[t, :3*e_d] = (y*(1-y)) * de_IFOGf[t, :3*e_d]
            
            # backprop matrix multiply
            de_WLSTM += np.outer(e_Hin[t], de_IFOG[t])
            de_Hin[t] = de_IFOG[t].dot(e_WLSTM.transpose())
      
            if t > 0: de_Hout[t-1] += de_Hin[t, 1+src_Ds.shape[1]:]
        
        return {'Wah':dWah, 'bah':dbah, 'WLSTM':dWLSTM, 'Wd':dWd, 'bd':dbd, 'e_WLSTM':de_WLSTM}
    