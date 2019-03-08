"""
Created on Jan 1st, 2019

An example script of running dialog sessions.

@author: zhangzthu
"""

from tasktk import Rule_Based_Multiwoz_Bot, User_Policy_Agenda_MultiWoz, DQN_Bot, Rule_Inform_Bot
from tasktk import Template_NLG #, lstm_nl
from tasktk import MDBT_Tracker
from tasktk import Dialog_System
from tasktk import User_Simulator
from tasktk import S2S_Simulator
from tasktk import Session
from tasktk import Log
from tasktk import joint_nlu
from tasktk import slot_dict, act_dict, dqn_params

import os
import tensorflow as tf


# demo setting
params = dict()
params['mode'] = 1
params['session_num'] = 100

params['nlg_model_path'] = './data/models/nlg/lstm_tanh_[1549590993.11]_24_28_1000_0.447.pkl'
params['nlu_model_path'] = './data/models/nlu/bi_lstm_[1549553112.36]_27_29_400_0.882.pkl'
#params['nlu_model_path'] = './data/nlg/bi_lstm_[1549553112.36]_27_29_400_0.882.p'


# TF setting
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
_config = tf.ConfigProto()
_config.gpu_options.allow_growth = True
_config.allow_soft_placement = True
global_sess = tf.Session(config=_config)

# components for system
sys_tracker = MDBT_Tracker(None, None, None)  # trainable tracker

saver = tf.train.Saver()
sys_tracker.restore_model(global_sess, saver)

#sys_policy = Rule_Based_Multiwoz_Bot(None, None, None)
sys_policy = Rule_Inform_Bot(None, None, None) # a simple inform rule agent

#sys_policy = DQN_Bot(act_dict, slot_dict, slot_dict, dqn_params)  # dqn policy

#sys_nlg = lstm_nlg(params['nlg_model_path']) # lstm nlg
# aggregate system components
#system_bot = Dialog_System(None, sys_tracker, sys_policy, sys_nlg, mode=params['mode'])
system_bot = Dialog_System(None, sys_tracker, sys_policy, None, mode=params['mode'])

# components for user
user_policy = User_Policy_Agenda_MultiWoz(None, None, None)
user_nlg = Template_NLG(None, None, None) # template nlg

#user_nlg = lstm_nlg(params['nlg_model_path']) # lstm nlg
#user_nlu = joint_nlu(params['nlu_model_path']) # joint_nlu

# aggregate user components
user_simulator = User_Simulator(None, None, user_policy, user_nlg, mode=params['mode'])

# session
session_controller = Session(system_bot, user_simulator)
logger = Log('session.txt')
logger.clear()
# run dialog

stat = {'success':0, 'fail':0}

for session_id in range(params['session_num']):
    session_over = False
    last_user_response = 'null'
    session_controller.init_session()
    session_controller.sess = global_sess
    
    print('******Episode %d******' % (session_id))
    print(user_simulator.policy.goal)

    while not session_over:
        system_response, user_response, session_over, reward = session_controller.next_turn(last_user_response)
        if not session_over:
            last_user_response = user_response
        logger.log('\tstate: {}'.format(system_bot.last_state['belief_state']['inform_slots']))
        logger.log('\tsystem: ' + '{}'.format(system_response))
        logger.log('\tuser: ' + '{}'.format(user_response))
        logger.log('\t-- turn end ---')
        
        # print to screen
        #print('\tstate: {}'.format(system_bot.last_state['current_slots']['inform_slots']))
        print('\tstate: {}'.format(system_bot.last_state.keys()))
        print('\tsystem: ' + '{}'.format(system_response))
        print('\tuser da: {}'.format(session_controller.simulator.current_action))
        print('\tuser: ' + '{}'.format(user_response))
        print('\t-- turn end ---')
        
    dialog_status = user_simulator.policy.goal.task_complete()
    if dialog_status:
        stat['success'] += 1
    else: stat['fail'] += 1
    
    print('task completion: {}'.format(user_simulator.policy.goal.task_complete()))   
    logger.log('---- session end ----')
    print('---- session end ----')
    # session_controller.train_sys()  # train the params of system agent

print('statistics: %s' % (stat))