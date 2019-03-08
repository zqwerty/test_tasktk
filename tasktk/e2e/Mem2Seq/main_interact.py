# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:34:18 2019

@author: truthless
"""

import numpy as np
import logging 
from tqdm import tqdm
import torch

from utils.config import *
from models.enc_vanilla import *
from models.enc_Luong import *
from models.enc_PTRUNK import *
from models.Mem2Seq import *

'''
python main_interact.py -dec=Mem2Seq -ds=woz -path=save/mem2seq-WOZ/[saved model dir]/
'''

BLEU = False

if (args['decoder'] == "Mem2Seq"):
    if args['dataset']=='kvr':
        from utils.utils_kvr_mem2seq import *
        BLEU = True
    elif args['dataset']=='woz':
        from utils.utils_woz_mem2seq import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi_mem2seq import *
    else: 
        print("You need to provide the --dataset information")
else:
    if args['dataset']=='kvr':
        from utils.utils_kvr import *
        BLEU = True
    elif args['dataset']=='babi':
        from utils.utils_babi import *
    else: 
        print("You need to provide the --dataset information")

# Configure models
directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0]

train, dev, test, testOOV, lang, max_len, max_r = prepare_data_seq(task, batch_size=1)

def plain2tensor(word2index, memory):
    src_seqs = []
    for token in memory:
        src_seq = []
        for word in token:
            if word in word2index:
                src_seq.append(word2index[word])
            else:
                src_seq.append(UNK_token)
        src_seqs.append([src_seq])
    return torch.LongTensor(src_seqs).cuda() if USE_CUDA else torch.LongTensor(src_seqs)

if args['decoder'] == "Mem2Seq":
    model = globals()[args['decoder']](
        int(HDD),max_len,max_r,lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0, unk_mask=0)
else:
    model = globals()[args['decoder']](
        int(HDD),max_len,max_r,lang,args['path'],task, lr=0.0, n_layers=int(L), dropout=0.0)

print('##########')
print('Start interaction.')
t = 0
memory = []
while True:
    usr = input('usr: ') 
    #example input: 'please find a restaurant called nusha .'
    if usr == 'END':
        break
    t += 1
    print('turn:', t)
    memory += generate_memory(usr, '$u', t)
    src_plain = (memory+[['$$$$']*MEM_TOKEN_SIZE],)
    src_seqs = plain2tensor(lang.word2index, src_plain[0])
    words = model.evaluate_batch(1, src_seqs, [len(src_plain[0])], None, None, None, None, src_plain)
    row = np.transpose(words)[0].tolist()
    if '<EOS>' in row:
        row = row[:row.index('<EOS>')]
    sys = ' '.join(row)
    print('sys:', sys)
    memory += generate_memory(sys, '$s', t)
