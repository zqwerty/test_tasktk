"""
Created on Jan 1st, 2019

@author: zhangzthu
"""

from tasktk.dst.mdbt_util import init_belief_state

class Dialog_System:
    """
    The bass class for a dialog agent which aggregates the four base components.
    """
    def __init__(self, nlu_model, tracker, policy, nlg_model, mode=0):
        """
        Args:
            nlu_model (NLU): An instance of NLU class.
            tracker (Tracker): An instance of Tracker class.
            policy (Sys_Policy): An instance of Policy class.
            nlg_model (NLG): An instance of NLG class.
            mode (int): 0 for utterance level and 1 for dialog act level.
        """
        self.nlu_model = nlu_model
        self.tracker = tracker
        self.policy = policy
        self.nlg_model = nlg_model
        self.mode = mode  # 0 for NL, 1 for DA level
        self.last_state = init_state()
        self.start = True

    def response(self, input, sess):
        """
        Generate the response of system bot.
        Args:
            input: Preorder user output, a 1) string if self.mode == 0, else 2) dialog act if self.mode == 1.
            sess (Session):
        Returns:
            output: Suystem response, a 1) string if self.mode == 0, else 2) dialog act if self.mode == 1.
        """
        # where is the input (user) processed? should upadte the state, then predict the agent action based on the state.
        
        if self.mode == 0:
            if self.start:
                self.start = False
                state = self.last_state
            else:
                state = self.tracker.update(self.last_state, sess)
            action = self.policy.predict(state)
            output = self.nlg_model.generate(action)
        elif self.mode == 1:
            if self.start:
                self.start = False
                state = self.last_state
            else:
                state = self.tracker.update(self.last_state, sess)
            action = self.policy.predict(state)
            output = action
        else:
            raise Exception('Unknown dialog mode: {}'.format(self.mode))
        self.last_state = state
        
        return output

    def init_session(self):
        """Init the parameters for a new session."""
        self.tracker.init_turn()
        self.policy.init_session()
        self.last_state = self.tracker.state
        self.start = True


def init_state():
    user_action = {}
    current_slots = init_belief_state
    current_slots['inform_slots'] = {}
    state = {'user_action': user_action,
             'belief_state': current_slots,
             'kb_result_dict': [],
             'history': []}
    return state