"""
Created on Jan 1st, 2019

@author: zhangzthu
"""

class User_Simulator:
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
        self.current_action = None
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
            # no state_tracker update?
            
            action, session_over, reward = self.policy.predict(None, input)  # see policy_agenda_multiwoz predict
            output = self.nlg_model.generate(action)  # templated nlg
        else:
            raise Exception('Unknown dialog mode: {}'.format(self.mode))
        # self.last_state = state
        
        #print('\tusr: ' + '{}'.format(action))
        #print('usr nl:', output)
        self.current_action = action
        return output, session_over, reward

    def init_session(self):
        """Init the parameters for a new session."""
        self.last_state = []
        self.policy.init_session()
        self.current_action = None