"""
Created on Jan 1st, 2019

The policy base class for system and user bot.

@author: zhangzthu
"""

class Sys_Policy:
    """Base class for system policy model."""

    def __init__(self, act_types, slots, slot_dict):
        """
        Constructor for Sys_Policy class.
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
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        pass

    def _build_model(self):
        """Build model for system policy."""
        pass

    def restore_model(self, model_path):
        """
        Restore
        Args:
            model_path (str): The path of model.
        """
        pass


class User_Policy:
    """Base model for user policy model."""
    def __init__(self, act_types, slots, slot_dict):
        """
        Constructor for User_Policy class.
        Args:
            act_types (list): A list of dialog acts.
            slots (list): A list of slot names.
            slot_dict (dict): Map slot name to its value set.
        """
        self.act_types = act_types
        self.slots = slots
        self.slot_dict = slot_dict

    def predict(self, state, sys_action):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
            sys_action (tuple): Preorder system action.s
        Returns:
            action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by the user.
        """
        pass
