from tasktk.policy.system.dqn_policy import DQN_Policy
from .dialog_config import *

class DQN_Bot(DQN_Policy):
    ''' DQN-based bot. '''
    
    def __init__(self, act_types=act_dict, slots=slot_dict, slot_val_dict=None, params=None):
        super().__init__(act_types, slots, slot_val_dict, params)
        

    def predict(self, state):
        """
        Args:
            State, please refer to util/state.py
        Output:
            DA(Dialog Act), in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """
        return super().predict(state)