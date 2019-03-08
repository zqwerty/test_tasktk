"""
Created on Jan 1st, 2019

@author: zhangzthu
"""

class Tracker:
    """Base class for dialog state tracker models."""
    def __init__(self, act_types, slots, slot_dict):
        """
        Args:
            act_types (list): The list of act types.
            slots (list): The list of slot names.
            slot_dict (dict): Map slot name to its value list.
        """
        self.act_types = act_types
        self.slots = slots
        self.slot_dict = slot_dict

    def update(self, previous_state, sess=None):
        """
        Update dialog state.
        Args:
            previous_state (tuple): Previous dialog state, which is a collection of distributions on the values of each
                    slot, in the same sequence of slots (__init__() parameter). For example, if slots = ['price',
                    cuisine] and slot_dict = {'price': ['cheap', 'expensive'], 'cuisine': ['chinese', 'british',
                    'french']}, and the current state is [cheap, british], then the ideal state variable is
                    (
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]
                    ).
                    Note that the extra one dimension indicates the slot is not mentioned yet.
            sess (Session Object):
        Returns:
            new_state (tuple): Updated dialog state, with the same form of previous state.
        """
        pass

class Rule_Tracker(Tracker):
    """The base class for rule-based DST methods, identical to Tracker class."""
    pass


class Trainable_Tracker(Tracker):
    """Base class for trainable DST models"""
    def __init__(self, act_types, slots, slot_dict):
        Tracker.__init__(self, act_types, slots, slot_dict)
        self.build_model()

    def build_model(self):
        """Build the model for DST."""
        pass

    def restore_model(self, sess, saver):
        """
        Restore model from model_path.
        Args:
            saver (tensorflow.python.training.saver.Saver):
        """
        pass

    def train(self):
        """Train the model."""
        pass

    def test(self, sess):
        """Test model."""
        pass