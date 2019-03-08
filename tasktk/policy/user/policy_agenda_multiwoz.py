#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   policy_agenda_multiwoz.py
@Time    :   2019/1/31 10:24
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :
"""

__time__ = '2019/1/31 10:24'

import random
import json
from tasktk.usr.goal_generator import GoalGenerator
from tasktk.policy.user.policy import User_Policy
from tasktk.policy.system.rule_based_multiwoz_bot import Rule_Based_Multiwoz_Bot, fake_state

DEF_VAL_UNK = '?'  # Unknown
DEF_VAL_DNC = 'not mentioned'  # Do not care
DEF_VAL_NUL = 'none'  # for none
DEF_VAL_BOOKED = 'yes'  # for booked
DEF_VAL_NOBOOK = 'no'  # for booked
NOT_SURE_VALS = [DEF_VAL_UNK, DEF_VAL_DNC, DEF_VAL_NUL, DEF_VAL_NOBOOK]

REF_USR_DA = {
    'Attraction': {
        'area': 'Area', 'type': 'Type', 'name': 'Name',
        'entrance fee': 'Fee', 'address': 'Addr',
        'postcode': 'Post', 'phone': 'Phone'
    },
    'Hospital': {
        'department': 'Department', 'address': 'Addr', 'postcode': 'Post',
        'phone': 'Phone'
    },
    'Hotel': {
        'type': 'Type', 'parking': 'Parking', 'pricerange': 'Price',
        'internet': 'Internet', 'area': 'Area', 'stars': 'Stars',
        'name': 'Name', 'stay': 'Stay', 'day': 'Day', 'people': 'People',
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Police': {
        'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'
    },
    'Restaurant': {
        'food': 'Food', 'pricerange': 'Price', 'area': 'Area',
        'name': 'Name', 'time': 'Time', 'day': 'Day', 'people': 'People',
        'phone': 'Phone', 'postcode': 'Post', 'address': 'Addr'
    },
    'Taxi': {
        'leaveAt': 'Leave', 'destination': 'Dest', 'departure': 'Depart', 'arriveBy': 'Arrive',
        'car type': 'Car', 'phone': 'Phone'
    },
    'Train': {
        'destination': 'Dest', 'day': 'Day', 'arriveBy': 'Arrive',
        'departure': 'Depart', 'leaveAt': 'Leave', 'people': 'People',
        'duration': 'Duration', 'price': 'Price', 'trainID': 'TrainID'
    }
}

REF_SYS_DA = {
    'Attraction': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Fee': "entrance fee", 'Name': "name", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Type': "type",
        'none': None, 'Open': None
    },
    'Hospital': {
        'Department': 'department', 'Addr': 'address', 'Post': 'postcode',
        'Phone': 'phone', 'none': None
    },
    'Booking': {
        'Day': 'day', 'Name': 'name', 'People': 'people',
        'Ref': 'ref', 'Stay': 'stay', 'Time': 'time',
        'none': None
    },
    'Hotel': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Internet': "internet", 'Name': "name", 'Parking': "parking",
        'Phone': "phone", 'Post': "postcode", 'Price': "pricerange",
        'Ref': "ref", 'Stars': "stars", 'Type': "type",
        'none': None
    },
    'Restaurant': {
        'Addr': "address", 'Area': "area", 'Choice': "choice",
        'Name': "name", 'Food': "food", 'Phone': "phone",
        'Post': "postcode", 'Price': "pricerange", 'Ref': "ref",
        'none': None
    },
    'Taxi': {
        'Arrive': "arriveBy", 'Car': "car type", 'Depart': "departure",
        'Dest': "destination", 'Leave': "leaveAt", 'Phone': "phone",
        'none': None
    },
    'Train': {
        'Arrive': "arriveBy", 'Choice': "choice", 'Day': "day",
        'Depart': "departure", 'Dest': "destination", 'Id': "trainID",
        'Leave': "leaveAt", 'People': "people", 'Ref': "ref",
        'Ticket': "ticket", 'Time': "duration", 'Duration': 'duration', 'none': None
    }
}

class User_Policy_Agenda_MultiWoz(User_Policy):
    """ The rule-based user policy model by agenda. Derived from the User_Policy class """

    def __init__(self, act_types, slots, slot_dict):
        """
        Constructor for User_Policy_Agenda class.
        Args:
            act_types (list): A list of dialog acts.
            slots (list): A list of slot names.
            slot_dict (dict): Map slot name to its value set.
        """
        self.max_turn = 20
        self.max_initiative = 4

        self.goal_generator = GoalGenerator(corpus_path='data/multiwoz/annotated_user_da_with_span_full.json')

        self.__turn = 0
        self.goal = None
        self.agenda = None

        User_Policy.__init__(self, act_types, slots, slot_dict)

    def init_session(self):
        """ Build new Goal and Agenda for next session """
        self.__turn = 0
        self.goal = Goal(self.goal_generator)
        self.agenda = Agenda(self.goal)

    def predict(self, state, sys_action):
        """
        Predict an user act based on state and preorder system action.
        Args:
            state (tuple): Dialog state.
            sys_action (tuple): Preorder system action.s
        Returns:
            action (tuple): User act.
            session_over (boolean): True to terminate session, otherwise session continues.
            reward (float): Reward given by user.
        """
        self.__turn += 2

        if self.__turn > self.max_turn:
            self.agenda.close_session()
        else:
            sys_action = self._transform_sysact_in(sys_action)
            self.agenda.update(sys_action, self.goal)
            if self.goal.task_complete():
                self.agenda.close_session()

        # A -> A' + user_action
        action = self.agenda.get_action(random.randint(1, self.max_initiative))

        # Is there any action to say?
        session_over = self.agenda.is_empty()

        # reward
        reward = self._reward()

        # transform to DA
        action = self._transform_usract_out(action)

        return action, session_over, reward

    def _transform_usract_out(self, action):
        new_action = {}
        for act in action.keys():
            if 'general' not in act:
                (dom, intent) = act.split('-')
                new_act = dom.capitalize() + '-' + intent.capitalize()
                new_action[new_act] = [[REF_USR_DA[dom.capitalize()].get(pairs[0], pairs[0]), pairs[1]] for pairs in action[act]]
            else:
                new_action[act] = action[act]

        return new_action

    def _transform_sysact_in(self, action):
        new_action = {}
        for act in action.keys():
            if 'general' not in act:
                (dom, _) = act.split('-')
                new_list = [[REF_SYS_DA[dom][pairs[0]], pairs[1]]
                            for pairs in action[act]
                            if REF_SYS_DA[dom].get(pairs[0], None) is not None]
                if len(new_list) > 0:
                    new_action[act.lower()] = new_list
            else:
                new_action[act] = action[act]

        return new_action

    def _reward(self):
        """
        Calculate reward based on task completion
        Returns:
            reward (float): Reward given by user.
        """
        if self.goal.task_complete():
            reward = 2.0 * self.max_turn
        elif self.agenda.is_empty():
            reward = -1.0 * self.max_turn
        else:
            reward = -1.0
        return reward


class Goal(object):
    """ User Goal Model Class. """

    def __init__(self, goal_generator: GoalGenerator):
        """
        create new Goal by random
        Args:
            goal_generator (GoalGenerator): Goal Gernerator.
        """
        self.domain_goals = goal_generator.get_user_goal()

        self.domains = list(self.domain_goals['domain_ordering'])
        del self.domain_goals['domain_ordering']

        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['reqt'] = {slot: DEF_VAL_UNK for slot in self.domain_goals[domain]['reqt']}

            if 'book' in self.domain_goals[domain].keys():
                self.domain_goals[domain]['booked'] = DEF_VAL_UNK

    def task_complete(self):
        """
        Check that all requests have been met
        Returns:
            (boolean): True to accomplish.
        """
        for domain in self.domains:
            if 'reqt' in self.domain_goals[domain]:
                requests = self.domain_goals[domain]['reqt']
                for slot in requests.keys():
                    if requests[slot] in NOT_SURE_VALS:
                        return False

            if 'booked' in self.domain_goals[domain]:
                if self.domain_goals[domain]['booked'] in NOT_SURE_VALS:
                    return False
        return True

    def __str__(self):
        return '-----Goal-----\n' + \
               json.dumps(self.domain_goals, indent=4) + \
               '\n-----Goal-----'


class Agenda(object):
    def __init__(self, goal: Goal):
        """
        Build a new agenda from goal
        Args:
            goal (Goal): User goal.
        """

        def random_sample(data, minimum=0, maximum=1000):
            return random.sample(data, random.randint(min(len(data), minimum), min(len(data), maximum)))

        self.CLOSE_ACT = 'general-bye'
        self.HELLO_ACT = 'general-greet'

        self.__stack = []

        # there is a 'bye' action at the bottom of the stack
        self.__push(self.CLOSE_ACT)

        for idx in range(len(goal.domains) - 1, -1, -1):
            domain = goal.domains[idx]
            # book
            if 'fail_book' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['fail_book'].keys(),
                                          len(goal.domain_goals[domain]['fail_book'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['fail_book'][slot])
            elif 'book' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['book'].keys(),
                                          len(goal.domain_goals[domain]['book'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['book'][slot])

            # request
            if 'reqt' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['reqt'].keys(),
                                          len(goal.domain_goals[domain]['reqt'])):
                    self.__push(domain + '-request', slot, DEF_VAL_UNK)

            # inform
            if 'fail_info' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['fail_info'].keys(),
                                          len(goal.domain_goals[domain]['fail_info'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['fail_info'][slot])
            elif 'info' in goal.domain_goals[domain]:
                for slot in random_sample(goal.domain_goals[domain]['info'].keys(),
                                          len(goal.domain_goals[domain]['info'])):
                    self.__push(domain + '-inform', slot, goal.domain_goals[domain]['info'][slot])

        if random.random() < 0.5:
            # Maybe this user is not a gentleman.
            self.__push(self.HELLO_ACT)

        self.cur_domain = None

    def update(self, sys_action, goal: Goal):
        """
        update Goal by current agent action and current goal. { A' + G" + sys_action -> A" }
        Args:
            sys_action (tuple): Preorder system action.s
            goal (Goal): User Goal
        """
        self._update_current_domain(sys_action, goal)

        for diaact in sys_action.keys():
            slot_vals = sys_action[diaact]
            if 'booking' in diaact:
                if self.update_booking(diaact, slot_vals, goal):
                    break
            elif 'general' in diaact:
                if self.update_general(diaact, slot_vals, goal):
                    break
            else:
                if self.update_domain(diaact, slot_vals, goal):
                    break

    def update_booking(self, diaact, slot_vals, goal: Goal):
        _, intent = diaact.split('-')
        domain = self.cur_domain

        if domain not in goal.domains:
            return False

        g_reqt = goal.domain_goals[domain].get('reqt', dict({}))
        g_info = goal.domain_goals[domain].get('info', dict({}))
        g_fail_info = goal.domain_goals[domain].get('fail_info', dict({}))
        g_book = goal.domain_goals[domain].get('book', dict({}))
        g_fail_book = goal.domain_goals[domain].get('fail_book', dict({}))

        if intent in ['book', 'inform']:
            info_right = True
            for [slot, value] in slot_vals:
                if slot in g_reqt:
                    if not self._check_item(domain + '-inform', slot):
                        self._remove_item(domain + '-request', slot)
                        g_reqt[slot] = value

                elif slot in g_fail_info and value != g_fail_info[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                    info_right = False
                elif len(g_fail_info) <= 0 and slot in g_info and value != g_info[slot]:
                    self._push_item(domain + '-inform', slot, g_info[slot])
                    info_right = False

                elif slot in g_fail_book and value != g_fail_book[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                    info_right = False
                elif len(g_fail_book) <= 0 and slot in g_book and value != g_book[slot]:
                    self._push_item(domain + '-inform', slot, g_book[slot])
                    info_right = False

                else:
                    pass

            if intent == 'book' and info_right:
                # booked ok
                if 'booked' in goal.domain_goals[domain]:
                    goal.domain_goals[domain]['booked'] = DEF_VAL_BOOKED
                self._push_item('general-thank')

        elif intent in ['nobook']:
            if len(g_fail_book) > 0:
                # Discard fail_book data and update the book data to the stack
                for slot in g_book.keys():
                    if (slot not in g_fail_book) or (slot in g_fail_book and g_fail_book[slot] != g_book[slot]):
                        self._push_item(domain + '-inform', slot, g_book[slot])

                # change fail_info name
                goal.domain_goals[domain]['fail_book_fail'] = goal.domain_goals[domain].pop('fail_book')
            elif 'booked' in goal.domain_goals[domain].keys():
                self.close_session()
                return True

        elif intent in ['request']:
            for [slot, _] in slot_vals:
                if slot in g_reqt:
                    pass
                elif slot in g_fail_info:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                elif len(g_fail_info) <= 0 and slot in g_info:
                    self._push_item(domain + '-inform', slot, g_info[slot])

                elif slot in g_fail_book:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                elif len(g_fail_book) <= 0 and slot in g_book:
                    self._push_item(domain + '-inform', slot, g_book[slot])

                else:

                    if domain == 'taxi' and (slot == 'destination' or slot == 'departure'):
                        places = [dom for dom in goal.domains[: goal.domains.index('taxi')] if
                                  'address' in goal.domain_goals[dom]['reqt']]

                        if len(places) >= 1 and slot == 'destination' and \
                                goal.domain_goals[places[-1]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-1]]['reqt']['address'])

                        elif len(places) >= 2 and slot == 'departure' and \
                                goal.domain_goals[places[-2]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-2]]['reqt']['address'])

                        else:
                            self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

                    else:
                        self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

        return False

    def update_general(self, diaact, slot_vals, goal: Goal):
        domain, intent = diaact.split('-')

        if intent == 'bye':
            self.close_session()
            return True
        elif intent == 'greet':
            pass
        elif intent == 'reqmore':
            pass
        elif intent == 'welcome':
            pass

        return False

    def update_domain(self, diaact, slot_vals, goal: Goal):
        domain, intent = diaact.split('-')

        if domain not in goal.domains:
            return False

        g_reqt = goal.domain_goals[domain].get('reqt', dict({}))
        g_info = goal.domain_goals[domain].get('info', dict({}))
        g_fail_info = goal.domain_goals[domain].get('fail_info', dict({}))
        g_book = goal.domain_goals[domain].get('book', dict({}))
        g_fail_book = goal.domain_goals[domain].get('fail_book', dict({}))

        if intent in ['inform', 'recommend', 'offerbook', 'offerbooked']:
            info_right = True
            for [slot, value] in slot_vals:
                if slot in g_reqt:
                    if not self._check_item(domain + '-inform', slot):
                        self._remove_item(domain + '-request', slot)
                        g_reqt[slot] = value

                elif slot in g_fail_info and value != g_fail_info[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                    info_right = False
                elif len(g_fail_info) <= 0 and slot in g_info and value != g_info[slot]:
                    self._push_item(domain + '-inform', slot, g_info[slot])
                    info_right = False

                elif slot in g_fail_book and value != g_fail_book[slot]:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                    info_right = False
                elif len(g_fail_book) <= 0 and slot in g_book and value != g_book[slot]:
                    self._push_item(domain + '-inform', slot, g_book[slot])
                    info_right = False

                else:
                    pass

            if intent == 'offerbooked' and info_right:
                # booked ok
                if 'booked' in goal.domain_goals[domain]:
                    goal.domain_goals[domain]['booked'] = DEF_VAL_BOOKED
                self._push_item('general-thank')

        elif intent in ['request']:
            for [slot, _] in slot_vals:
                if slot in g_reqt:
                    pass
                elif slot in g_fail_info:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                elif len(g_fail_info) <= 0 and slot in g_info:
                    self._push_item(domain + '-inform', slot, g_info[slot])

                elif slot in g_fail_book:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                elif len(g_fail_book) <= 0 and slot in g_book:
                    self._push_item(domain + '-inform', slot, g_book[slot])

                else:

                    if domain == 'taxi' and (slot == 'destination' or slot == 'departure'):
                        places = [dom for dom in goal.domains[: goal.domains.index('taxi')] if
                                  'address' in goal.domain_goals[dom]['reqt']]

                        if len(places) >= 1 and slot == 'destination' and \
                                goal.domain_goals[places[-1]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-1]]['reqt']['address'])

                        elif len(places) >= 2 and slot == 'departure' and \
                                goal.domain_goals[places[-2]]['reqt']['address'] not in NOT_SURE_VALS:
                            self._push_item(domain + '-inform', slot, goal.domain_goals[places[-2]]['reqt']['address'])

                        else:
                            self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

                    else:
                        self._push_item(domain + '-inform', slot, DEF_VAL_DNC)

        elif intent in ['nooffer']:
            if len(g_fail_info) > 0:
                # update all requests
                for slot in g_reqt.keys():
                    if g_reqt[slot] in NOT_SURE_VALS:
                        self._push_item(domain + '-request', slot, DEF_VAL_UNK)

                # Discard fail_info data and update the info data to the stack
                for slot in g_info.keys():
                    if (slot not in g_fail_info) or (slot in g_fail_info and g_fail_info[slot] != g_info[slot]):
                        self._push_item(domain + '-inform', slot, g_info[slot])

                # change fail_info name
                goal.domain_goals[domain]['fail_info_fail'] = goal.domain_goals[domain].pop('fail_info')
            elif len(g_reqt.keys()) > 0:
                self.close_session()
                return True

        elif intent in ['select']:
            # delete Choice
            slot_vals = [[slot, val] for [slot, val] in slot_vals if slot != 'choice']

            if len(slot_vals) > 0:
                slot = slot_vals[0][0]

                if slot in g_fail_info:
                    self._push_item(domain + '-inform', slot, g_fail_info[slot])
                elif len(g_fail_info) <= 0 and slot in g_info:
                    self._push_item(domain + '-inform', slot, g_info[slot])

                elif slot in g_fail_book:
                    self._push_item(domain + '-inform', slot, g_fail_book[slot])
                elif len(g_fail_book) <= 0 and slot in g_book:
                    self._push_item(domain + '-inform', slot, g_book[slot])

                else:
                    [slot, value] = random.choice(slot_vals)
                    self._push_item(domain + '-inform', slot, value)

                    if slot in g_reqt:
                        self._remove_item(domain + '-request', slot)
                        g_reqt[slot] = value

        return False

    def close_session(self):
        """ Clear up all actions """
        self.__stack = []
        self.__push(self.CLOSE_ACT)

    def get_action(self, initiative=1):
        """
        get multiple acts based on initiative
        Args:
            initiative (int): number of slots , just for 'inform'
        Returns:
            action (dict): user diaact
        """
        diaacts, slots, values = self.__pop(initiative)
        action = {}
        for (diaact, slot, value) in zip(diaacts, slots, values):
            if diaact not in action.keys():
                action[diaact] = []
            action[diaact].append([slot, value])

        return action

    def is_empty(self):
        """
        Is the agenda already empty
        Returns:
            (boolean): True for empty, False for not.
        """
        return len(self.__stack) <= 0

    def _update_current_domain(self, sys_action, goal: Goal):
        for diaact in sys_action.keys():
            domain, _ = diaact.split('-')
            if domain in goal.domains:
                self.cur_domain = domain

    def _remove_item(self, diaact, slot=DEF_VAL_UNK):
        for idx in range(len(self.__stack)):
            if 'general' in diaact:
                if self.__stack[idx]['diaact'] == diaact:
                    self.__stack.remove(self.__stack[idx])
                    break
            else:
                if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                    self.__stack.remove(self.__stack[idx])
                    break

    def _push_item(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self._remove_item(diaact, slot)
        self.__push(diaact, slot, value)

    def _check_item(self, diaact, slot):
        for idx in range(len(self.__stack)):
            if self.__stack[idx]['diaact'] == diaact and self.__stack[idx]['slot'] == slot:
                return True
        return False

    def __check_next_diaact(self):
        if len(self.__stack) > 0:
            return self.__stack[-1]['diaact']
        return None

    def __push(self, diaact, slot=DEF_VAL_NUL, value=DEF_VAL_NUL):
        self.__stack.append({'diaact': diaact, 'slot': slot, 'value': value})

    def __pop(self, initiative=1):
        diaacts = []
        slots = []
        values = []

        for _ in range(initiative):
            try:
                item = self.__stack.pop(-1)
                diaacts.append(item['diaact'])
                slots.append(item['slot'])
                values.append(item['value'])

                if self.__check_next_diaact() == self.CLOSE_ACT:
                    break
            except:
                break

        return diaacts, slots, values

    def __str__(self):
        text = '\n-----agenda-----\n'
        text += '<stack top>\n'
        for item in reversed(self.__stack):
            text += str(item) + '\n'
        text += '<stack btm>\n'
        text += '-----agenda-----\n'
        return text


def test():
    user_simulator = User_Policy_Agenda_MultiWoz(None, None, None)
    user_simulator.init_session()

    test_turn(user_simulator, {"Hotel-Inform": [["Type", "qqq"], ["Parking", "no"]]})
    test_turn(user_simulator, {"Hotel-Request": [["Addr", "?"]], "Hotel-Inform": [["Internet", "yes"]]})
    test_turn(user_simulator, {"Hotel-Nooffer": [["Stars", "3"]], "Hotel-Request": [["Parking", "?"]]})
    test_turn(user_simulator, {"Hotel-Select": [["Area", "aa"], ["Area", "bb"], ["Area", "cc"], ['Choice', 3]]})
    test_turn(user_simulator, {"Hotel-Offerbooked": [["Ref", "12345"]]})
    test_turn(user_simulator, {"Booking-Nobook": [["Area", "aaa"]]})


def test_turn(user_simulator, sys_action):
    action, session_over, reward = user_simulator.predict(None, sys_action)

    print('----------------------------------')
    print('sys_action :' + str(sys_action))
    print('user_action:' + str(action))
    print('over       :' + str(session_over))
    print('reward     :' + str(reward))
    print(user_simulator.goal)
    print(user_simulator.agenda)


def test_with_system():
    user_simulator = User_Policy_Agenda_MultiWoz(None, None, None)
    user_simulator.init_session()
    state = fake_state()
    system_agent = Rule_Based_Multiwoz_Bot(None, None, None)
    sys_action = system_agent.predict(state)
    action, session_over, reward = user_simulator.predict(None, sys_action)
    print("Sys:")
    print(json.dumps(sys_action, indent=4))
    print("User:")
    print(json.dumps(action, indent=4))


if __name__ == '__main__':
    test()
    # test_with_system()
