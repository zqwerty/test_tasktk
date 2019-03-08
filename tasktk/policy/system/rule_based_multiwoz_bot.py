from tasktk.policy.system.policy_rule_based import Rule_Based_Sys_Policy
import random
import json

REF_SYS_DA = {
    'Attraction': {'address': 'Addr', 'area': 'Area', 'choice': 'Choice', 'entrance fee': 'Fee', 'name': 'Name',
                   'phone': 'Phone', 'postcode': 'Post', 'pricerange': 'Price', 'type': 'Type'},
    'Hospital': {'department': 'Department', 'address': 'Addr', 'postcode': 'Post', 'phone': 'Phone'},
    'Booking': {'day': 'Day', 'name': 'Name', 'people': 'People', 'ref': 'Ref', 'stay': 'Stay', 'time': 'Time'},
    'Hotel': {'address': 'Addr', 'area': 'Area', 'choice': 'Choice', 'internet': 'Internet', 'name': 'Name',
              'parking': 'Parking', 'phone': 'Phone', 'postcode': 'Post', 'pricerange': 'Price', 'ref': 'Ref',
              'stars': 'Stars', 'type': 'Type'},
    'Restaurant': {'address': 'Addr', 'area': 'Area', 'choice': 'Choice', 'name': 'Name', 'food': 'Food',
                   'phone': 'Phone', 'postcode': 'Post', 'pricerange': 'Price', 'ref': 'Ref'},
    'Taxi': {'arriveBy': 'Arrive', 'car type': 'Car', 'departure': 'Depart', 'destination': 'Dest', 'leaveAt': 'Leave',
             'phone': 'Phone'},
    'Train': {'arriveBy': 'Arrive', 'choice': 'Choice', 'day': 'Day', 'departure': 'Depart', 'destination': 'Dest',
              'trainID': 'Id', 'leaveAt': 'Leave', 'people': 'People', 'ref': 'Ref', 'ticket': 'Ticket',
              'duration': 'Duration'}}

# Information required to finish booking, according to different domain.
booking_info = {'Train': ['People'],
                'Restaurant': ['Time', 'Day', 'People'],
                'Hotel': ['Stay', 'Day', 'People']}

# Alphabet used to generate Ref number
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Judge if user has confirmed a unique choice, according to different domain
token = {'Attraction': ['Name', 'Addr', ''],
         'Hotel': ['Name', ]}


class Rule_Based_Multiwoz_Bot(Rule_Based_Sys_Policy):
    ''' Rule-based bot. Implemented for Multiwoz dataset. '''

    def predict(self, state):
        """
        Args:
            State, please refer to util/state.py
        Output:
            DA(Dialog Act), in the form of {act_type1: [[slot_name_1, value_1], [slot_name_2, value_2], ...], ...}
        """
        # print('policy received state: {}'.format(state))
        DA = {}

        for user_act in state['user_action']:
            domain, intent_type = user_act.split('-')

            # Respond to general greetings
            if domain == 'general':
                self._update_greeting(user_act, state, DA)

            # Book taxi for user
            elif domain == 'Taxi':
                self._book_taxi(user_act, state, DA)

            # User's talking about other domain
            elif domain != "Train":
                self._update_DA(user_act, state, DA)

            # Info about train
            else:
                self._update_train(user_act, state, DA)

            # Judge if user want to book
            self._judge_booking(user_act, state, DA)

            if 'Booking-Book' in DA:
                if random.random() < 0.5:
                    DA['general-reqmore'] = []

        return DA

    # General request / inform
    def _update_greeting(self, user_act, state, DA):
        _, intent_type = user_act.split('-')

        # Respond to goodbye
        if intent_type == 'bye':
            if 'general-bye' not in DA:
                DA['general-bye'] = []
            if random.random() < 0.3:
                if 'general-welcome' not in DA:
                    DA['general-welcome'] = []
        elif intent_type == 'thank':
            DA['general-welcome'] = []
            if random.random() < 0.8:
                DA['general-bye'] = []

    # Book a taxi for user
    def _book_taxi(self, user_act, state, DA):
        blank_info = []
        for info in state['belief_state']['taxi']['semi']:
            if state['belief_state']['taxi']['semi'][info] == "":
                info = REF_SYS_DA['Taxi'].get(info[0], info[0])
                blank_info.append(info)

        # Finish booking, tell user car type and phone number
        if len(blank_info) == 0:
            if 'Taxi-Inform' not in DA:
                DA['Taxi-Inform'] = []
            car = generate_car()
            phone_num = generate_ref_num(11)
            DA['Taxi-Inform'].append(['Car', car])
            DA['Taxi-Inform'].append(['Phone', phone_num])
            return

        # Need essential info to finish booking
        request_num = random.randint(0, 999999) % len(blank_info) + 1
        if 'Taxi-Request' not in DA:
            DA['Taxi-Request'] = []
        for i in range(request_num):
            slot = REF_SYS_DA.get(blank_info[i], blank_info[i])
            DA['Taxi-Request'].append([slot, '?'])

    # Answer user's utterance about any domain other than taxi or train
    def _update_DA(self, user_act, state, DA):

        domain, intent_type = user_act.split('-')

        # Respond to user's request
        if intent_type == 'Request':
            if (domain + "-Inform") not in DA:
                DA[domain + "-Inform"] = []
            for slot in state['user_action'][user_act]:
                slot_name = REF_SYS_DA[domain].get(slot[0], slot[0])
                DA[domain + "-Inform"].append([slot_name, state['kb_results_dict'][0][slot[0].lower()]])

        else:
            # There's no result matching user's constraint
            if len(state['kb_results_dict']) == 0:
                if (domain + "-NoOffer") not in DA:
                    DA[domain + "-NoOffer"] = []

                for slot in state['belief_state'][domain.lower()]['semi']:
                    if state['belief_state'][domain.lower()]['semi'][slot] != "" and \
                            state['belief_state'][domain.lower()]['semi'][slot] != "don't care":
                        slot_name = REF_SYS_DA[domain].get(slot, slot)
                        DA[domain + "-NoOffer"].append([slot_name, state['belief_state'][domain.lower()]['semi'][slot]])

                p = random.random()

                # Ask user if he wants to change constraint
                if p < 0.3:
                    req_num = random.randint(0, 999999) % len(DA[domain + "-NoOffer"]) + 1
                    if [domain + "-Request"] not in DA:
                        DA[domain + "-Request"] = []
                    for i in range(req_num):
                        print(DA[domain + "-NoOffer"])
                        slot_name = REF_SYS_DA[domain].get(DA[domain + "-NoOffer"][i][0], DA[domain + "-NoOffer"][i][0])
                        DA[domain + "-Request"].append([slot_name, "?"])

            # There's exactly one result matching user's constraint
            elif len(state['kb_results_dict']) == 1:

                # Inform user about this result
                if (domain + "-Inform") not in DA:
                    DA[domain + "-Inform"] = []
                props = []
                for prop in state['belief_state'][domain.lower()]['semi']:
                    props.append(prop)
                property_num = len(props)
                info_num = random.randint(0, 999999) % property_num + 1
                random.shuffle(props)
                for i in range(info_num):
                    slot_name = REF_SYS_DA[domain].get(props[i], props[i])
                    DA[domain + "-Inform"].append([slot_name, state['kb_results_dict'][0][props[i]]])

            # There are multiple resultes matching user's constraint
            else:
                p = random.random()

                # Recommend a choice from kb_list
                if p < 0.2:
                    if (domain + "-Choice") not in DA:
                        DA[domain + "-Choice"] = []
                    DA[domain + "-Choice"].append(["Choice", len(state['kb_results_dict'])])
                    idx = random.randint(0, 999999) % len(state['kb_results_dict'])
                    choice = state['kb_results_dict'][idx]
                    DA[domain + "-Choice"].append(['name', choice['name']])
                    props = []
                    for prop in choice:
                        props.append([prop, choice[prop]])
                    prop_num = min(random.randint(0, 999999) % 3, len(props))
                    random.shuffle(props)
                    for i in range(prop_num):
                        slot = props[i][0]
                        string = slot[0].upper() + slot[1:]
                        string = REF_SYS_DA.get(string, string)
                        DA[domain + "-Choice"].append([string, props[i][1]])

                # Ask user to choose a candidate.
                elif p < 0.3:
                    prop_values = []
                    props = []
                    for prop in state['kb_results_dict'][0]:
                        for candidate in state['kb_results_dict']:
                            if candidate[prop] not in prop_values:
                                prop_values.append(candidate[prop])
                        if len(prop_values) > 1:
                            props.append([prop, prop_values])
                        prop_values = []
                    random.shuffle(props)
                    if domain + "-Choice" not in DA:
                        DA[domain + "-Choice"] = []
                    for i in range(len(props[0][1])):
                        print(props[0])
                        prop_value = REF_SYS_DA[domain].get(props[0][0], props[0][0])
                        DA[domain + "-Choice"].append([prop_value, props[0][1][i]])

                # Ask user for more constraint
                else:
                    reqs = []
                    for prop in state['belief_state'][domain.lower()]['semi']:
                        if state['belief_state'][domain.lower()]['semi'][prop] == "":
                            prop_value = REF_SYS_DA[domain].get(prop, prop)
                            reqs.append([prop_value, "?"])
                    req_num = random.randint(0, 999999) % len(reqs) + 1
                    random.shuffle(reqs)
                    if (domain + "-Request") not in DA:
                        DA[domain + "-Request"] = []
                    for i in range(req_num):
                        req = reqs[i]
                        req[0] = REF_SYS_DA[domain].get(req[0], req[0])
                        DA[domain + "-Request"].append(req)

    def _update_train(self, user_act, state, DA):
        if len(state['kb_results_dict']) == 0:
            if "Train-NoOffer" not in DA:
                DA['Train-NoOffer'] = []
            return

        if 'Train-Request' not in DA:
            DA['Train-Request'] = []
        for prop in state['belief_state']['train']['semi']:
            if state['belief_state']['train']['semi'][prop] == "":
                slot = REF_SYS_DA['Train'].get(prop, prop)
                DA["Train-Request"].append([slot, '?'])

    # If user want to book, return a ref number
    def _judge_booking(self, user_act, state, DA):
        print(user_act)
        domain, _ = user_act.split('-')
        for slot in state['belief_state']:
            if domain in booking_info and slot in booking_info[domain]:
                if 'Booking-Book' not in DA:
                    DA['Booking-Book'] = generate_ref_num(8)


# Generate a ref num for booking.
def generate_ref_num(length):
    string = ""
    while len(string) < length:
        string += alphabet[random.randint(0, 999999) % 36]
    return string


# Generate a car for taxi booking
def generate_car():
    car_types = ["toyota", "skoda", "bmw", "honda", "ford", "audi", "lexus", "volvo", "volkswagen", "tesla"]
    p = random.randint(0, 999999) % len(car_types)
    return car_types[p]


def fake_state():
    user_action = {'Hotel-Request': [['Name', '?']], 'Train-Inform': [['Day', 'don\'t care']]}
    init_belief_state = {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    }
    current_slots = {'inform_slots': None}
    current_slots['inform_slots'] = {'price': 'cheap', 'people': '15', 'day': 'tuesday', 'dest': 'cam'}
    kb_results = [None, None]
    kb_results[0] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south'}
    kb_results[1] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'north'}
    state = {'user_action': user_action,
             'belief_state': init_belief_state,
             'kb_results_dict': kb_results,
             'hotel-request': [['phone']]}
    '''
    state = {'user_action': dict(),
             'belief_state: dict(),
             'kb_results_dict': kb_results
    }
    '''
    return state


def init_state():
    user_action = ['general-hello']
    current_slots = dict()
    current_slots['inform_slots'] = {}
    kb_results = [None, None]
    kb_results[0] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'south'}
    kb_results[1] = {'name': 'xxx_train', 'day': 'tuesday', 'dest': 'cam', 'phone': '123-3333', 'area': 'north'}
    state = {'user_action': user_action,
             'current_slots': current_slots,
             'kb_results_dict': []}
    return state


def test_run():
    policy = Rule_Based_Multiwoz_Bot(None, None, None)
    system_act = policy.predict(fake_state())
    print(json.dumps(system_act, indent=4))



class Rule_Inform_Bot(Rule_Based_Sys_Policy):
    """ a simple, inform rule bot """
    
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
        
        self.cur_inform_slot_id = 0
        self.cur_request_slot_id = 0
        self.domains = ['Taxi']
    
    def init_session(self):
        """
        Restore after one session
        """
        self.cur_inform_slot_id = 0
        self.cur_request_slot_id = 0
        
    def predict(self, state):
        
        act_slot_response = {}
        domain = self.domains[0]        
            
        if self.cur_inform_slot_id < len(REF_SYS_DA[domain]):
            key = list(REF_SYS_DA[domain])[self.cur_inform_slot_id]
            slot = REF_SYS_DA[domain][key]
            
            diaact = domain + "-Inform"
            val = generate_car()
                    
            act_slot_response[diaact] = []
            act_slot_response[diaact].append([slot, val])
            
            self.cur_inform_slot_id += 1
        elif self.cur_request_slot_id < len(REF_SYS_DA[domain]):
            key = list(REF_SYS_DA[domain])[self.cur_request_slot_id]
            slot = REF_SYS_DA[domain][key]
            
            diaact = domain + "-Request"
            val = "?"
                    
            act_slot_response[diaact] = []
            act_slot_response[diaact].append([slot, val])
            
            self.cur_request_slot_id += 1
        else:
            act_slot_response['general-bye'] = []
            self.cur_request_slot_id = 0
            self.cur_inform_slot_id = 0
            
        return act_slot_response


if __name__ == '__main__':
    test_run()
