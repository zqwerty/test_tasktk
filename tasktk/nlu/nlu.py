"""
Created on Jan 1st, 2019

@author: zhangzthu
"""

from .error import Error_NLU

class NLU:
    """Base class for NLU model."""

    def __init__(self, act_types, slots, slot_dict, act_type_rate=0, slot_rate=0):
        """
        Constructor for NLU class.
        Args:
            act_types (list): A list of dialog acts, e.g., ['inform', 'request', ...]
            slots (list): A list of slot names. e.g., ['price', 'location', 'cuisine', ...]
            slot_dict (dict): Map slot name to its value list.
            act_type_rate (float): Act type level error rate, where 0.0 <= act_type_rate <= 1.0.
            slot_rate (float): Slot level error rate, 0.0 <= slot_rate <= 1.0.
        """
        self.act_types = act_types
        self.slots = slots
        self.slot_dict = slot_dict
        self.error_model = Error_NLU(act_type_rate, slot_rate)

    def predict(self, utterance):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (tuple): Dialog act with noise.
                    For example, for an utterance input 'i want a cheap place', the output dialog act should be:
                    ('inform', {'price'='cheap'}). After applying noise, the result can be different.
        """
        dialog_act = self._predict(utterance)
        return self.error_model.apply(dialog_act)

    def _predict(self, utterance):
        """
        Predict the dialog act of a natural language utterance.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (tuple): Dialog act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        pass


class Rule_NLU(NLU):
    """Base class for rule-based NLU model, identical to NLU class."""
    pass


class Trainable_NLU(NLU):
    """Base class for trainable NLU model."""

    def __init__(self, act_types, slots, slot_dict):
        NLU.__init__(self, act_types, slots, slot_dict)
        self.build_model()

    def build_model(self):
        """
        Build NLU model.
        Returns:

        """
        pass

    def restore_model(self, model_path):
        """
        Restore model from model_path.
        Args:
            model_path (str): The path of model file.
        Returns:

        """
        pass

    def train(self):
        """
        Train the dialog model.
        Returns:

        """
        pass

    def test(self):
        """
        Test the trained dialog model.
        Returns:

        """
        pass