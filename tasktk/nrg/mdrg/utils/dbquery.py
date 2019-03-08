"""
Created on Feb 10th, 2019

@author: sungjinl 
"""
import os
import random
import json
import numpy as np


# loading databases
domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital', 'taxi', 'police']
dbs = {}
for domain in domains:
    dbs[domain] = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'db/{}_db.json'.format(domain))))

def query(domain, constraints, ignore_open=True):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    if domain == 'taxi':
        return [{'taxi_colors': random.choice(dbs[domain]['taxi_colors']), 
        'taxi_types': random.choice(dbs[domain]['taxi_types']), 
        'taxi_phone': [random.randint(1, 9) for _ in range(10)]}]
    if domain == 'police':
        return dbs['police']
    if domain == 'hospital':
        return dbs['hospital']

    found = []
    for record in dbs[domain]:
        for key, val in constraints:
            if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
                pass
            else:
                if key not in record:
                    continue
                if key == 'leaveAt':
                    val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                    val2 = int(record['leaveAt'].split(':')[0]) * 100 + int(record['leaveAt'].split(':')[1])
                    if val1 > val2:
                        break
                elif key == 'arriveBy':
                    val1 = int(val.split(':')[0]) * 100 + int(val.split(':')[1])
                    val2 = int(record['arriveBy'].split(':')[0]) * 100 + int(record['arriveBy'].split(':')[1])
                    if val1 < val2:
                        break
                elif ignore_open and key in ['destination', 'departure', 'name']:
                    continue
                else:
                    if val.strip() != record[key].strip():
                        break
        else:
            found.append(record)

    return found 


def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0,0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector
