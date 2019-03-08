from tasktk.policy.policy import Sys_Policy


class Rule_Based_Sys_Policy(Sys_Policy):
	''' Rule-based system policy. Derived from Sys_Policy class.'''

	def __init__(self, act_types, slots, slot_dict):
		"""
        Constructor for Rule_Based_Sys_Policy class.
        Args:
            act_types (list): A list of dialog acts.
            slots (list): A list of slot names.
            slot_dict (dict): Map slot name to its value set.
        """
		self.act_types = act_types
		self.slots = slots
		self.slot_dict = slot_dict
		self._build_model()

	def predict(self, state):
		"""
        Predict an system action given state.
        Args:
            state (dict): Please check util/state.py
        Returns:
            action (list): System act, in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """
		pass
	
	def init_session(self):
		"""
        Restore after one session
        """
		pass