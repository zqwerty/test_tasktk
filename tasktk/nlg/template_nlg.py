from tasktk.nlg.nlg import Rule_NLG

class Template_NLG(Rule_NLG):
    def init(self, act_types, slots, slot_dict):
        Rule_NLG.__init__(self, act_types, slots, slot_dict)

    def generate(self, dialog_act):
        phrases = []
        for da in dialog_act.keys():
            domain, type = da.split('-')
            if domain == 'general':
                if type == 'hello':
                    phrases.append('hello, i need help')
                else:
                    phrases.append('bye')
            elif type == 'Request':
                for slot, value in dialog_act[da]:
                    phrases.append('what is the {}'.format(slot))
            else:
                for slot, value in dialog_act[da]:
                    phrases.append('i want the {} to be {}'.format(slot, value))
        sent = ', '.join(phrases)
        return sent


if __name__ == '__main__':
    nlg = Template_NLG(None, None, None)
    user_acts = [{"Restaurant-Inform": [["Food", "japanese"], ["Time", "17:45"]]},
                 {"Restaurant-Request": [["Price", "?"]]},
                 {"general-bye": [["none", "none"]]}]
    for ua in user_acts:
        sent = nlg.generate(ua)
        print(sent)
