'''
Created on Feb. 14, 2019

@author: xiul
'''
import os



start_dia_acts = {
    #'greeting':[],
    'request':['moviename', 'starttime', 'theater', 'city', 'state', 'date', 'genre', 'ticket', 'numberofpeople']
}

movie_request_slots = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
movie_inform_slots = ['moviename', 'theater'] #, 'starttime'



dqn_params = {}
dqn_params['max_turn'] = 20
dqn_params['epsilon'] = 0.0
dqn_params['experience_replay_pool_size'] = 1000
dqn_params['dqn_hidden_size'] = 60
dqn_params['batch_size'] = 16
dqn_params['gamma'] = 0.9
dqn_params['predict_mode'] = False
dqn_params['simulation_epoch_size'] = 100
dqn_params['warm_start'] = 1
dqn_params['warm_start_epochs'] = 100

dqn_params['trained_model_path'] = None
dqn_params['agent_act_level'] = 0
dqn_params['agent_run_mode'] = 1
dqn_params['write_model_dir'] = './ckpts/'
dqn_params['save_check_point'] = 10

dqn_params['success_rate_threshold'] = 0.3
    
################################################################################
# Dialog status
################################################################################
FAILED_DIALOG = -1
SUCCESS_DIALOG = 1
NO_OUTCOME_YET = 0

# Rewards
SUCCESS_REWARD = 50
FAILURE_REWARD = 0
PER_TURN_REWARD = 0

################################################################################
#  Special Slot Values
################################################################################
I_DO_NOT_CARE = "I do not care"
NO_VALUE_MATCH = "NO VALUE MATCHES!!!"
TICKET_AVAILABLE = 'Item Available'

################################################################################
#  Constraint Check
################################################################################
CONSTRAINT_CHECK_FAILURE = 0
CONSTRAINT_CHECK_SUCCESS = 1

################################################################################
#  NLG Beam Search
################################################################################
nlg_beam_size = 5

################################################################################
#  run_mode: 0 for dia-act; 1 for NL; 2 for no output
################################################################################
run_mode = 0
auto_suggest = 0

################################################################################
#   A Basic Set of Feasible actions to be Consdered By an RL agent
################################################################################
feasible_actions = [
    ############################################################################
    #   greeting actions
    ############################################################################
    {'diaact':"general-greet", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"general-thanks", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   thanks actions
    ############################################################################
    {'diaact':"general-bye", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   general-welcome actions
    ############################################################################
    {'diaact':"general-welcome", 'inform_slots':{}, 'request_slots':{}},
    ############################################################################
    #   general-reqmore actions
    ############################################################################
    {'diaact':"general-reqmore", 'inform_slots':{}, 'request_slots':{}}
]

slot_dict = []
act_dict = []

def load_feasible_actions(file_path):
    """ load feasible actions """
    
    file = open(file_path, 'r')
    lines = [line.strip().strip('\n').strip('\r') for line in file]
    
    diaact = None
    for lid, line in enumerate(lines):
        arr = line.split('\t')
        #print(lid, line, len(arr))
        
        feasible_action = {'diaact':"", 'inform_slots':{}, 'request_slots':{}}
        feasible_action['diaact'] = arr[0]
        
        if len(arr) == 1: 
            diaact = arr[0].strip()
            continue
        elif len(arr) == 2:
            slot = arr[1].strip()
            if slot == 'none': pass
            else:
                if 'Request' in diaact:
                    feasible_action['request_slots'][slot] = 'UNK'
                else:
                    feasible_action['inform_slots'][slot] = 'PLACEHOLDER'
                    
                if slot not in slot_dict:
                    slot_dict.append(slot)
                    #slot_dict[slot] = len(slot_dict)
        
        if diaact not in act_dict:
            #act_dict[diaact] = len(act_dict)
            act_dict.append(diaact)
            
        feasible_action['diaact'] = diaact
        feasible_actions.append(feasible_action)
        #print(lid, line, len(arr), feasible_action)
        
        
# load feasible actions
load_feasible_actions(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data/multiwoz/dialog_act_slot.txt'))
#'./data/multiwoz/dialog_act_slot.txt')
print("# feasible actions: %d" % (len(feasible_actions)))
print("# diaact: {}".format(len(act_dict)))
print("# slot: {}".format(len(slot_dict)))

############################################################################
#   Adding the inform actions
############################################################################
#for slot in sys_inform_slots:
#    feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

############################################################################
#   Adding the request actions
############################################################################
#for slot in sys_request_slots:
#    feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})



    