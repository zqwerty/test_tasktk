"""
Created on Jan 1st, 2019

@author: zhangzthu
"""

class Session:
    """
    A dialog controller which aggregates two agents to conduct a complete dialog session or to train the dialog policy
    model.
    """
    def __init__(self, sys_agent, simulator):
        """
        Constructor for Session class.
        Args:
            sys_agent (Dialog_System): An instance of dialog systenm agent.
            simulator (User_Simulator): An instance of user simulator.
        """
        self.sys_agent = sys_agent
        self.simulator = simulator
        self.sys_agent.init_session()
        self.simulator.init_session()
        self.sess = None

    def next_turn(self, last_user_response):
        """
        Conduct a new turn of dialog, which consists of the system response and user response.
        Note that the system response comes first in each turn, and in the 1st turn its response can be like "hello, how
        can I help you".
        The variable type of responses can be either 1) str or 2) dialog act, depends on the dialog mode settings of the
        two agents which are supposed to be the same.
        Args:
            last_user_response: Last user response.
        Returns:
            sys_response: The response of system.
            user_response:The response of user simulator.
            session_over (boolean): True if session ends, else session continues.
            reward (float): The reward given by the user.
        """
        system_response = self.sys_agent.response(last_user_response, self.sess)
        user_response, session_over, reward = self.simulator.response(system_response)
        str_sys_response = 'this is a system response sentence placeholder'
        # save the new responses into state history
        self.sys_agent.last_state['history'].append([str_sys_response, user_response])

        # the act type list of user action
        # self.sys_agent.last_state['user_action'] = list(self.simulator.current_action.keys())
        self.sys_agent.last_state['user_action'] = self.simulator.current_action
        # the complete user act of this turn
        self.sys_agent.last_state['user_act_full'] = self.simulator.current_action

        return system_response, user_response, session_over, reward

    def train_sys(self):
        """
        Train the parameters of system agent.
        """
        self.sys_agent.policy.train()

    def init_session(self):
        self.sys_agent.init_session()
        self.simulator.init_session()

