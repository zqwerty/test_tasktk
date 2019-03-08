from tasktk.dst.state_tracker import Trainable_Tracker
import tensorflow as tf
from tasktk.dst.mdbt_util import model_definition, load_word_vectors, load_ontology, load_woz_data, init_state, \
        track_dialogue, generate_batch, process_history, evaluate_model
import os, sys, json, math, time
import numpy as np
import copy
from random import shuffle
from tasktk.util.kb_query import KB_Query
from tasktk.dst.mdbt_util import init_belief_state

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data/mdbt')
VALIDATION_URL = os.path.join(DATA_PATH, "data/validate.json")
WORD_VECTORS_URL = os.path.join(DATA_PATH, "word-vectors/paragram_300_sl999.txt")
TRAINING_URL = os.path.join(DATA_PATH, "data/train.json")
ONTOLOGY_URL = os.path.join(DATA_PATH, "data/ontology.json")
TESTING_URL = os.path.join(DATA_PATH, "data/test.json")
MODEL_URL = os.path.join(DATA_PATH, "models/model-1")
GRAPH_URL = os.path.join(DATA_PATH, "graphs/graph-1")
RESULTS_URL = os.path.join(DATA_PATH, "results/log-1.txt")
KB_URL = os.path.join(DATA_PATH, "data/")  # TODO: yaoqin
TRAIN_MODEL_URL = os.path.join(DATA_PATH, "train_models/model-1")
TRAIN_GRAPH_URL = os.path.join(DATA_PATH, "train_graph/graph-1")

#ROOT_URL = '../../data/mdbt'

#VALIDATION_URL = "./data/mdbt/data/validate.json"
#WORD_VECTORS_URL = "./data/mdbt/word-vectors/paragram_300_sl999.txt"
#TRAINING_URL = "./data/mdbt/data/train.json"
#ONTOLOGY_URL = "./data/mdbt/data/ontology.json"
#TESTING_URL = "./data/mdbt/data/test.json"
#MODEL_URL = "./data/mdbt/models/model-1"
#GRAPH_URL = "./data/mdbt/graphs/graph-1"
#RESULTS_URL = "./data/mdbt/results/log-1.txt"
#KB_URL = "./data/mdbt/data/"  # TODO: yaoqin
#TRAIN_MODEL_URL = "./data/mdbt/train_models/model-1"
#TRAIN_GRAPH_URL = "./data/mdbt/train_graph/graph-1"

domains = ['restaurant', 'hotel ', 'attraction', 'train', 'taxi']

train_batch_size = 1
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0

num_slots = 0

booking_slots = {}


class MDBT_Tracker(Trainable_Tracker):
    """
    A multi-domain belief tracker, adopted from https://github.com/osmanio2/multi-domain-belief-tracking.
    """
    def __init__(self, act_types, slots, slot_dict):
        Trainable_Tracker.__init__(self, act_types, slots, slot_dict)

        self.word_vectors = load_word_vectors(WORD_VECTORS_URL)

        # Load the ontology and extract the feature vectors
        self.ontology, self.ontology_vectors, self.slots = load_ontology(ONTOLOGY_URL, self.word_vectors)

        # Load and process the training data
        self.dialogues, self.actual_dialogues = load_woz_data(TESTING_URL, self.word_vectors, self.ontology)
        self.no_dialogues = len(self.dialogues)

        self.model_variables = model_definition(self.ontology_vectors, len(self.ontology), self.slots, num_hidden=None,
                                                bidir=True, net_type=None, test=True, dev='cpu')
        self.state = init_state()
        self.model_url = MODEL_URL
        self.graph_url = GRAPH_URL
        self.kb_query = KB_Query('')

    def init_turn(self):
        self.state = init_state()

    def update(self, prev_state, sess=None):
        """Update the dialog state."""
        #if not os.path.exists("../../data/mdbt/results"):
        #    os.makedirs("../../data/mdbt/results")
        if not os.path.exists(os.path.join(DATA_PATH, "results")):
            os.makedirs(os.path.join(DATA_PATH, "results"))

        global train_batch_size, MODEL_URL, GRAPH_URL

        model_variables = self.model_variables
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, predictions,
         true_predictions, [y, _]) = model_variables

        # generate fake dialogue based on history (this os to reuse the original MDBT code)
        actual_history = prev_state['history']  # [[sys, user], [sys, user], ...]
        # print(actual_history)
        fake_dialogue = {}
        turn_no = 0
        for _sys, _user in actual_history:
            turn = {}
            turn['system'] = _sys
            fake_user = {}
            fake_user['text'] = _user
            fake_user['belief_state'] = init_belief_state
            turn['user'] = fake_user
            key = str(turn_no)
            fake_dialogue[key] = turn
            turn_no += 1
        context, actual_context = process_history([fake_dialogue], self.word_vectors, self.ontology)

        # generate turn input
        batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
                batch_no_turns = generate_batch(context, 0, 1, len(self.ontology))  # old feature

        # run model
        [pred, y_pred] = sess.run(
            [predictions, y],
            feed_dict={user: batch_user, sys_res: batch_sys,
                       labels: batch_labels,
                       domain_labels: batch_domain_labels,
                       user_uttr_len: batch_user_uttr_len,
                       sys_uttr_len: batch_sys_uttr_len,
                       no_turns: batch_no_turns,
                       keep_prob: 1.0})

        # convert to str output
        dialgs, _, _ = track_dialogue(actual_context, self.ontology, pred, y_pred)
        assert len(dialgs) >= 1
        last_turn = dialgs[0][-1]
        predictions = last_turn['prediction']
        new_belief_state = copy.deepcopy(init_belief_state)
        current_slots_inform = copy.deepcopy(prev_state['belief_state']['inform_slots'])
        current_slots = copy.deepcopy(prev_state['belief_state'])
        for item in predictions:
            domain, slot, value = item.strip().split('-')
            value = value[::-1].split(':', 1)[1][::-1]
            if slot == 'price range':
                slot = 'pricerange'
            if slot not in ['name', 'book']:
                current_slots_inform[slot] = value
                if domain in current_slots:  # update current_slots
                    domain_pairs = current_slots[domain]
                    if 'semi' in domain_pairs and slot in domain_pairs['semi']:
                        current_slots[domain]['semi'][slot] = value
            if slot in new_belief_state[domain]['semi']:
                new_belief_state[domain]['semi'][slot] = value
            elif 'book' in slot:
                try:
                    book_slot = slot.strip().split()[1]
                    if book_slot in new_belief_state[domain]['book']:
                        new_belief_state[domain]['book'][book_slot] = value
                except:
                    print("book without slot value:", slot)
            else:
                print("Cannot handle the item in preditions:", item)
        # update slot
        # add dict() func. to avoid error
        # print(prev_state)
        new_state = copy.deepcopy(dict(prev_state))
        # new_state['current_slots'] = {'inform_slots': current_slots_inform}
        new_state['belief_state'] = current_slots
        new_state['belief_state']['inform_slots'] = current_slots_inform
        # issue kb query using updated state
        kb_result_dict = self.kb_query.query(new_state)
        new_state['kb_results_dict'] = kb_result_dict
        new_state['belief_state'] = new_belief_state
        return new_state

    def restore_model(self, sess, saver):
        saver.restore(sess, self.model_url)
        print('\tMDBT: mdbt model restored from ', self.model_url)

    def train(self):
        """
            Train the model.
            Model saved to
        """
        num_hid, bidir, net_type, n2p, batch_size, model_url, graph_url, dev = \
                None, True, None, None, None, None, None, None
        global train_batch_size, MODEL_URL, GRAPH_URL, device, TRAIN_MODEL_URL, TRAIN_GRAPH_URL

        if batch_size:
            train_batch_size = batch_size
            print("Setting up the batch size to {}.........................".format(batch_size))
        if model_url:
            TRAIN_MODEL_URL = model_url
            print("Setting up the model url to {}.........................".format(TRAIN_MODEL_URL))
        if graph_url:
            TRAIN_GRAPH_URL = graph_url
            print("Setting up the graph url to {}.........................".format(TRAIN_GRAPH_URL))

        if dev:
            device = dev
            print("Setting up the device to {}.........................".format(device))

        # 1 Load and process the input data including the ontology
        # Load the word embeddings
        word_vectors = load_word_vectors(WORD_VECTORS_URL)

        # Load the ontology and extract the feature vectors
        ontology, ontology_vectors, slots = load_ontology(ONTOLOGY_URL, word_vectors)

        # Load and process the training data
        dialogues, _ = load_woz_data(TRAINING_URL, word_vectors, ontology)
        no_dialogues = len(dialogues)

        # Load and process the validation data
        val_dialogues, _ = load_woz_data(VALIDATION_URL, word_vectors, ontology)

        # Generate the validation batch data
        val_data = generate_batch(val_dialogues, 0, len(val_dialogues), len(ontology))
        val_iterations = int(len(val_dialogues) / train_batch_size)

        # 2 Initialise and set up the model graph
        # Initialise the model
        graph = tf.Graph()
        with graph.as_default():
            model_variables = model_definition(ontology_vectors, len(ontology), slots, num_hidden=num_hid, bidir=bidir,
                                               net_type=net_type, dev=device)
            (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
             slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, _, _, _) = model_variables
            [precision, recall, value_f1] = value_f1
            saver = tf.train.Saver()
            if device == 'gpu':
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})

            sess = tf.Session(config=config)
            if os.path.exists(TRAIN_MODEL_URL + ".index"):
                saver.restore(sess, TRAIN_MODEL_URL)
                print("Loading from an existing model {} ....................".format(TRAIN_MODEL_URL))
            else:
                if not os.path.exists(TRAIN_MODEL_URL):
                    os.makedirs('/'.join(TRAIN_MODEL_URL.split('/')[:-1]))
                    os.makedirs('/'.join(TRAIN_GRAPH_URL.split('/')[:-1]))
                init = tf.global_variables_initializer()
                sess.run(init)
                print("Create new model parameters.....................................")
            merged = tf.summary.merge_all()
            val_accuracy = tf.summary.scalar('validation_accuracy', value_accuracy)
            val_f1 = tf.summary.scalar('validation_f1_score', value_f1)
            train_writer = tf.summary.FileWriter(TRAIN_GRAPH_URL, graph)
            train_writer.flush()

        # 3 Perform an epoch of training
        last_update = -1
        best_f_score = -1
        for epoch in range(no_epochs):

            batch_size = train_batch_size
            sys.stdout.flush()
            iterations = math.ceil(no_dialogues / train_batch_size)
            start_time = time.time()
            val_i = 0
            shuffle(dialogues)
            for batch_id in range(iterations):

                if batch_id == iterations - 1 and no_dialogues % iterations != 0:
                    batch_size = no_dialogues % train_batch_size

                batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
                batch_no_turns = generate_batch(dialogues, batch_id, batch_size, len(ontology))

                [_, summary, da, sa, va, vf, pr, re] = sess.run([train_step, merged, domain_accuracy, slot_accuracy,
                                                                 value_accuracy, value_f1, precision, recall],
                                                                feed_dict={user: batch_user, sys_res: batch_sys,
                                                                           labels: batch_labels,
                                                                           domain_labels: batch_domain_labels,
                                                                           user_uttr_len: batch_user_uttr_len,
                                                                           sys_uttr_len: batch_sys_uttr_len,
                                                                           no_turns: batch_no_turns,
                                                                           keep_prob: 0.5})

                print("The accuracies for domain is {:.2f}, slot {:.2f}, value {:.2f}, f1_score {:.2f} precision {:.2f}"
                      " recall {:.2f} for batch {}".format(da, sa, va, vf, pr, re, batch_id + iterations * epoch))

                train_writer.add_summary(summary, start_batch + batch_id + iterations * epoch)

                # ================================ VALIDATION ==============================================

                if batch_id % batches_per_eval == 0 or batch_id == 0:
                    if batch_id == 0:
                        print("Batch", "0", "to", batch_id, "took", round(time.time() - start_time, 2), "seconds.")

                    else:
                        print("Batch", batch_id + iterations * epoch - batches_per_eval, "to",
                              batch_id + iterations * epoch, "took",
                              round(time.time() - start_time, 3), "seconds.")
                        start_time = time.time()

                    _, _, v_acc, f1_score, sm1, sm2 = evaluate_model(sess, model_variables, val_data,
                                                                     [val_accuracy, val_f1], batch_id, val_i)
                    val_i += 1
                    val_i %= val_iterations
                    train_writer.add_summary(sm1, start_batch + batch_id + iterations * epoch)
                    train_writer.add_summary(sm2, start_batch + batch_id + iterations * epoch)
                    stime = time.time()
                    current_metric = f1_score
                    print(" Validation metric:", round(current_metric, 5), " eval took",
                          round(time.time() - stime, 2), "last update at:", last_update, "/", iterations)

                    # and if we got a new high score for validation f-score, we need to save the parameters:
                    if current_metric > best_f_score:
                        last_update = batch_id + iterations * epoch + 1
                        print("\n ====================== New best validation metric:", round(current_metric, 4),
                              " - saving these parameters. Batch is:", last_update, "/", iterations,
                              "---------------- ===========  \n")

                        best_f_score = current_metric

                        saver.save(sess, TRAIN_MODEL_URL)

            print("The best parameters achieved a validation metric of", round(best_f_score, 4))

    def test(self, sess):
        """Test the MDBT model. Almost the same as original code."""
        if not os.path.exists("../../data/mdbt/results"):
            os.makedirs("../../data/mdbt/results")

        global train_batch_size, MODEL_URL, GRAPH_URL

        model_variables = self.model_variables
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, predictions,
         true_predictions, [y, _]) = model_variables
        [precision, recall, value_f1] = value_f1
        # print("\tMDBT: Loading from an existing model {} ....................".format(MODEL_URL))

        iterations = math.ceil(self.no_dialogues / train_batch_size)
        batch_size = train_batch_size
        [slot_acc, tot_accuracy] = [np.zeros(len(self.ontology), dtype="float32"), 0]
        slot_accurac = 0
        # value_accurac = np.zeros((len(slots),), dtype="float32")
        value_accurac = 0
        joint_accuracy = 0
        f1_score = 0
        preci = 0
        recal = 0
        processed_dialogues = []
        # np.set_printoptions(threshold=np.nan)
        for batch_id in range(int(iterations)):

            if batch_id == iterations - 1:
                batch_size = self.no_dialogues - batch_id * train_batch_size

            batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
            batch_no_turns = generate_batch(self.dialogues, batch_id, batch_size, len(self.ontology))

            [da, sa, va, vf, pr, re, pred, true_pred, y_pred] = sess.run(
                [domain_accuracy, slot_accuracy, value_accuracy,
                 value_f1, precision, recall, predictions,
                 true_predictions, y],
                feed_dict={user: batch_user, sys_res: batch_sys,
                           labels: batch_labels,
                           domain_labels: batch_domain_labels,
                           user_uttr_len: batch_user_uttr_len,
                           sys_uttr_len: batch_sys_uttr_len,
                           no_turns: batch_no_turns,
                           keep_prob: 1.0})

            true = sum([1 if np.array_equal(pred[k, :], true_pred[k, :]) and sum(true_pred[k, :]) > 0 else 0
                        for k in range(true_pred.shape[0])])
            actual = sum([1 if sum(true_pred[k, :]) > 0 else 0 for k in range(true_pred.shape[0])])
            ja = true / actual
            tot_accuracy += da
            # joint_accuracy += ja
            slot_accurac += sa
            if math.isnan(pr):
                pr = 0
            preci += pr
            recal += re
            if math.isnan(vf):
                vf = 0
            f1_score += vf
            # value_accurac += va
            slot_acc += np.mean(np.asarray(np.equal(pred, true_pred), dtype="float32"), axis=0)

            dialgs, va1, ja = track_dialogue(self.actual_dialogues[batch_id * train_batch_size:
                                             batch_id * train_batch_size + batch_size],
                                             self.ontology, pred, y_pred)
            processed_dialogues += dialgs
            joint_accuracy += ja
            value_accurac += va1

            print(
                "The accuracies for domain is {:.2f}, slot {:.2f}, value {:.2f}, other value {:.2f}, f1_score {:.2f} precision {:.2f}"
                " recall {:.2f}  for batch {}".format(da, sa, np.mean(va), va1, vf, pr, re, batch_id))

        print(
            "End of evaluating the test set...........................................................................")

        slot_acc /= iterations
        # print("The accuracies for each slot:")
        # print(value_accurac/iterations)
        print("The overall accuracies for domain is"
              " {}, slot {}, value {}, f1_score {}, precision {},"
              " recall {}, joint accuracy {}".format(tot_accuracy / iterations, slot_accurac / iterations,
                                                     value_accurac / iterations, f1_score / iterations,
                                                     preci / iterations, recal / iterations,
                                                     joint_accuracy / iterations))

        with open(RESULTS_URL, 'w') as f:
            json.dump(processed_dialogues, f, indent=4)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    _config.allow_soft_placement = True
    global_sess = tf.Session(config=_config)
    start_time = time.time()
    mdbt = MDBT_Tracker(None, None, None)
    print('\tMDBT: model build time: {:.2f} seconds'.format(time.time()-start_time))
    saver = tf.train.Saver()
    mdbt.restore_model(global_sess, saver)
    # demo state
    _state = init_state()
    _state['history'] = [['null', 'I\'m trying to find an expensive restaurant in the centre part of town.'],
                         ['The Cambridge Chop House is an good expensive restaurant in the centre of town. Would you like me to book it for you?',
                          'Yes, a table for 1 at 16:15 on sunday.  I need the reference number.']]
    new_state = mdbt.update(_state, global_sess)
    print(json.dumps(new_state, indent=4))
    print('all time: {:.2f} seconds'.format(time.time() - start_time))
