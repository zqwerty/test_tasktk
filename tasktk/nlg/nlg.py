"""
Created on Jan 1st, 2019

@author: zhangzthu
"""

class NLG:
    """Base class for NLG model."""
    def __init__(self, act_types, slots, slot_dict):
        """
        Constructor for NLG class.
        Args:
            act_types (list): A list of dialog acts.
            slots (list): A list of slot names.
            slot_dict (dict): Map slot name to its value set.
        """
        self.act_types = act_types
        self.slots = slots
        self.slot_dict = slot_dict

    def generate(self, dialog_act):
        """
        Generate a natural language utterance conditioned on the dialog act.
        Args:
            dialog_act (tuple): Dialog act, with the form of (act_type, {slot_name_1: value_1,
                    slot_name_2, value_2, ...})
        Returns:
            response (str): A natural langauge utterance.
        """
        pass


class Rule_NLG(NLG):
    """Base class for rule-based NLG model, identical to NLG class."""
    
    def load_predefined_templates(self, file_path):
        """ load pre_defined nlg templates """
        pass


class Trainable_NLG(NLG):
    """Base class for trainable NLG model."""
    
    def __init__(self, act_set, slot_set, slot_dict):
        NLG.__init__(self, act_set, slot_set, slot_dict)
        self.build_model()

    def build_model(self):
        """Build NLG model."""
        pass

    def restore_model(self, model_path):
        """
        Restore model from model_path.
        Args:
            model_path (str): The path of model file.
        """
        pass

    def train(self):
        """Train the nlg model."""
        pass

    def test(self):
        """Test the trained nlg model."""
        pass