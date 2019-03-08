"""
template NLG for multiwoz dataset. templates are in `multiwoz_template_nlg/` dir.
See `example` function in this file for usage.
"""
import json
import random
import os


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


slot2word = {
    'Fee': 'fee',
    'Addr': 'address',
    'Area': 'area',
    'Stars': 'stars',
    'Internet': 'Internet',
    'Department': 'department',
    'Choice': 'choice',
    'Ref': 'reference number',
    'Food': 'food',
    'Type': 'type',
    'Price': 'price range',
    'Stay': 'stay',
    'Phone': 'phone',
    'Post': 'postcode',
    'Day': 'day',
    'TrainID': 'TrainID',
    'Name': 'name',
    'Car': 'car type',
    'Leave': 'leave',
    'Time': 'time',
    'Arrive': 'arrive',
    'Ticket': 'ticket',
    'Depart': 'departure',
    'People': 'people',
    'Dest': 'destination',
    'Parking': 'parking',
    'Duration': 'duration',
    'Open': 'open',
    'Id': 'Id'
}


class MultiwozTemplateNLG:
    def __init__(self):
        """
        both template are dict, *_template[dialog_act][slot] is a list of templates.
        """
        template_dir = os.path.dirname(os.path.abspath(__file__))
        self.auto_user_template = read_json(os.path.join(template_dir, 'auto_user_template_nlg.json'))
        self.auto_system_template = read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
        self.manual_user_template = read_json(os.path.join(template_dir, 'manual_user_template_nlg.json'))
        self.manual_system_template = read_json(os.path.join(template_dir, 'manual_system_template_nlg.json'))

    def generate(self, dialog_acts, is_user=True, mode="auto"):
        """
        NLG for Multiwoz dataset
        :param dialog_acts: {da1:[[slot1,value1],...], da2:...}
        :param is_user: if dialog_act from user or system
        :param mode:    `auto`: templates extracted from data without manual modification, may have no match;
                        `manual`: templates with manual modification, sometimes verbose;
                        `auto_manual`: use auto templates first. When fails, use manual templates.
        :return: generated sentence (if no match, return 'None')
        """
        if mode=='manual':
            if is_user:
                template = self.manual_user_template
            else:
                template = self.manual_system_template

            return self._manual_generate(dialog_acts, template)

        elif mode=='auto':
            if is_user:
                template = self.auto_user_template
            else:
                template = self.auto_system_template

            return self._auto_generate(dialog_acts, template)

        elif mode=='auto_manual':
            if is_user:
                template1 = self.auto_user_template
                template2 = self.manual_user_template
            else:
                template1 = self.auto_system_template
                template2 = self.manual_system_template

            res = self._auto_generate(dialog_acts, template1)
            if res == 'None':
                res = self._manual_generate(dialog_acts, template2)
            return res

        else:
            raise Exception("Invalid mode! available mode: auto, manual, auto_manual")

    def _postprocess(self,sen):
        sen = sen.strip().capitalize()
        if sen[-1] != '?' and sen[-1] != '.':
            sen += '.'
        sen += ' '
        return sen

    def _manual_generate(self, dialog_acts, template):
        sentences = ''
        for dialog_act, slot_value_pairs in dialog_acts.items():
            if 'Select' in dialog_act:
                slot2values = {}
                for slot, value in slot_value_pairs:
                    slot2values.setdefault(slot, [])
                    slot2values[slot].append(value)
                for slot, values in slot2values.items():
                    if slot == 'none': continue
                    sentence = 'Do you prefer ' + values[0]
                    for i, value in enumerate(values[1:]):
                        if i == (len(values) - 2):
                            sentence += ' or ' + value
                        else:
                            sentence += ' , ' + value
                    sentence += ' {} ? '.format(slot2word[slot])
                    sentences += sentence
            elif 'Request' in dialog_act:
                for slot, value in slot_value_pairs:
                    if slot not in template[dialog_act]:
                        sentence = 'What is the {} of {} ? '.format(slot, dialog_act.split('-')[0].lower())
                        sentences += sentence
                    else:
                        sentence = random.choice(template[dialog_act][slot])
                        sentence = self._postprocess(sentence)
                        sentences += sentence
            elif 'general' in dialog_act:
                sentence = random.choice(template[dialog_act]['none'])
                sentence = self._postprocess(sentence)
                sentences += sentence
            else:
                for slot, value in slot_value_pairs:
                    if slot in template[dialog_act]:
                        sentence = random.choice(template[dialog_act][slot])
                        sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), slot.upper()), value)
                    else:
                        sentence = 'The {} is {} . '.format(slot2word[slot], value)
                    sentence = self._postprocess(sentence)
                    sentences += sentence
        return sentences.strip()

    def _auto_generate(self, dialog_acts, template):
        sentences = ''
        for dialog_act, slot_value_pairs in dialog_acts.items():
            key = ''
            for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                key += s + ';'
            if key in template[dialog_act]:
                sentence = random.choice(template[dialog_act][key])
                if 'Request' in dialog_act or 'general' in dialog_act:
                    sentence = self._postprocess(sentence)
                    sentences += sentence
                else:
                    for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                        if v != 'none':
                            sentence = sentence.replace('#{}-{}#'.format(dialog_act.upper(), s.upper()), v, 1)
                    sentence = self._postprocess(sentence)
                    sentences += sentence
            else:
                return 'None'
        return sentences.strip()


def example():
    # dialog act
    dialog_acts = {'Train-Inform': [['Day', 'wednesday'], ['Leave', '10:15']]}
    # whether from user or system
    is_user = False

    multiwoz_template_nlg = MultiwozTemplateNLG()
    print(dialog_acts)
    print(multiwoz_template_nlg.generate(dialog_acts, is_user, mode='manual'))
    print(multiwoz_template_nlg.generate(dialog_acts, is_user, mode='auto'))
    print(multiwoz_template_nlg.generate(dialog_acts, is_user, mode='auto_manual'))


if __name__ == '__main__':
    example()
