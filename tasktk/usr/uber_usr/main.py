import data_loader
import Config
from model import e2e_user
import argparser
import tensorflow as tf

config = Config.config
input_graph = tf.Graph()
with input_graph.as_default():
    data = data_loader().DataLoader()
    proto = tf.ConfigProto()
    input_sess = tf.Session(config=proto)

    seq_goals, seq_usr_dass, seq_sys_dass = data.data_loader()
    train_goals, train_usrdas, train_sysdas, test_goals, test_usrdas, test_sysdas, val_goals, val_usrdas, val_sysdas = train_test_val_split(
        seq_goals, seq_usr_dass, seq_sys_dass)
    generator = batch_iter(train_goals, train_usrdas, train_sysdas)

def train():
    best_ppl = 1e20


def infer():
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda x: x.lower() == 'true')
    parser.add_argument("--train", type="bool", default=True)
    args = parser.parse_args(sys.argv[1:])
    return args

def main():
    args = get_args()

    if args.train:
        for batch_goals, batch_usrdas, batch_sysdas in generator:
            train()
    else:
        infer()

if __name__ == "__main__":
    main()
