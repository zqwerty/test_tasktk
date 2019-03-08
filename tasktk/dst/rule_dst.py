from tasktk.dst import Rule_Tracker

import copy

class Rule_DST(Rule_Tracker):
    """Rule based DST which trivially updates new values from NLU result to states."""
    def __init__(self, act_types, slots, slot_dict):
        Rule_Tracker.__init__(act_types, slots, slot_dict)

    def update(self, previous_state, sess=None):
        """
            Simply update new value from NLU results.
            Note that the inform slot-value pairs of NLU result is in
                    previous_state['user_da']['inform'][-1]
        """
        user_da = previous_state['user_da'][-1]
        new_inform_slots = user_da['inform'].keys()
        current_slots_inform = copy.deepcopy(previous_state['current_slots']['inform_slots'])
        # current_slots = copy.deepcopy(previous_state['current_slots'])
        for slot in new_inform_slots:
            current_slots_inform[slot] = new_inform_slots['slot']

        new_state = copy.deepcopy(previous_state)
        new_state['belief_state']['inform_slots'] = current_slots_inform
        kb_result_dict = self.kb_query.query(new_state)
        new_state['kb_result_dict'] = kb_result_dict
        return new_state
