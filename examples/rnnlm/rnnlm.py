import argparse
from collections import defaultdict
import logging
import sys

import numpy as np

import minn
import minn.contrib.functions as F


class Vocab(object):

    def __init__(self):
        self.w2i = defaultdict(lambda: len(self.w2i))

    def add(self, word):
        return self.w2i[word]

    def get(self, word, default=None):
        return self.w2i[word] if word in self.w2i else default

    def __getitem__(self, word):
        return self.get(word, None)

    def __len__(self):
        return len(self.w2i)


class Loader(object):

    def __init__(self, unknown_word='<UNK>'):
        self.vocab = Vocab()
        self.unk_id = self.vocab.add(unknown_word)
        self.bos_id = self.vocab.add('<BOS>')
        self.eos_id = self.vocab.add('<EOS>')

    def load(self, file, train=False):
        def preprocess(x):
            return x.lower()

        def map(words):
            if train:
                return [self.vocab.add(preprocess(w)) for w in words]
            else:
                return [self.vocab.get(preprocess(w), self.unk_id)
                        for w in words]

        reader = self._gen_reader(file)
        sentences = [np.array(map(words), np.int32) for words in reader]
        return sentences

    @staticmethod
    def _gen_reader(file):
        for line in open(file, mode='r'):
            line = line.strip()
            if not line:
                continue
            line = '<BOS> ' + line + ' <EOS>'
            words = line.split()
            yield words


class RNNLanguageModel(minn.Model):

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.E = minn.Parameter((embed_dim, vocab_size))
        self.U = minn.Parameter((hidden_dim, embed_dim))
        self.W = minn.Parameter((hidden_dim, hidden_dim))
        self.b1 = minn.Parameter((hidden_dim,))
        self.V = minn.Parameter((vocab_size, hidden_dim))
        self.b2 = minn.Parameter((vocab_size,))
        self._vocab_size = vocab_size

    def initialize(self):
        initializer = minn.initializers.NormalInitializer()
        self.E.initialize(initializer)
        self.U.initialize(initializer)
        self.W.initialize(initializer)
        self.b1.initialize(0.)
        self.V.initialize(initializer)
        self.b2.initialize(0.)

    def forward(self, x):
        E = F.parameter(self.E)
        U = F.parameter(self.U)
        W = F.parameter(self.W)
        b1 = F.parameter(self.b1)
        V = F.parameter(self.V)
        b2 = F.parameter(self.b2)

        xp = minn.get_device().xp
        s = F.input(xp.zeros((1, U.shape[0]), dtype=xp.float32))
        y = []
        for x_t in x:
            onehot = xp.zeros((1, self._vocab_size), dtype=xp.float32)
            onehot[:, x_t] = 1.
            w = F.input(onehot) @ E.T
            s = F.sigmoid(w @ U.T + s @ W.T + b1)
            y_t = s @ V.T + b2
            y.append(y_t)
        return y

    def loss(self, y, t):
        assert len(y) == len(t)
        t = t.reshape((t.size, 1))
        loss = 0.0
        for y_t, t_t in zip(y, t):
            loss += F.softmax_cross_entropy(y_t, t_t)
        return loss


def get_batches(xs, batch_size, pad_id, shuffle=False):
    num_samples = len(xs)
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    offset = 0
    while offset < num_samples:
        samples = np.take(xs, indices[offset:offset + batch_size], axis=0)
        max_len = max(len(x) for x in samples)
        batch = np.full((len(samples), max_len), pad_id, dtype=np.int32)
        for i, x in enumerate(samples):
            batch[i, :x.size] = x
        offset += batch_size
        yield batch


def train(train_file,
          valid_file=None,
          epochs=20,
          batch_size=32,
          learning_rate=0.001):
    loader = Loader()
    train_sents = loader.load(train_file, train=True)
    valid_sents = None
    if valid_file:
        valid_sents = loader.load(valid_file, train=False)
    eos_id = loader.eos_id

    model = RNNLanguageModel(
        vocab_size=len(loader.vocab),
        embed_dim=100,
        hidden_dim=64)
    model.initialize()
    optimizer = minn.optimizers.SGD(learning_rate)
    optimizer.add(model)

    def process(sentences, train):
        epoch_loss = 0.0
        processed = 0
        num_samples = len(sentences)
        for batch in get_batches(sentences, batch_size, eos_id, shuffle=train):
            minn.clear_graph()
            x, t = batch[:-1].T, batch[1:].T
            y = model.forward(x)
            loss = model.loss(y, t)
            epoch_loss += loss.data * len(batch)
            if train:
                optimizer.reset_gradients()
                loss.backward()
                optimizer.update()
            processed += len(batch)
            sys.stderr.write("{} / {} = {:.2f}%\r".format(
                processed, num_samples, (processed / num_samples) * 100))
            sys.stderr.flush()
        print("", file=sys.stderr)
        ppl = np.exp(float(epoch_loss) / sum(len(s) - 1 for s in sentences))
        return ppl

    do_validation = valid_sents is not None

    for epoch in range(1, epochs + 1):
        ppl = process(train_sents, True)
        logging.info(
            "[{}] epoch {} - #samples: {}, ppl: {:.8f}"
            .format("train", epoch, len(train_sents), ppl))
        if do_validation:
            ppl = process(valid_sents, False)
            logging.info(
                "[{}] epoch {} - #samples: {}, ppl: {:.8f}"
                .format("valid", epoch, len(valid_sents), ppl))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        metavar='FILE', dest='train_file', help='training data file')
    parser.add_argument(
        '--valid', metavar='FILE', dest='valid_file',
        help='validation data file')
    parser.add_argument(
        '--lr', metavar='RATE', type=float, default=0.001,
        dest='learning_rate', help='learning rate')
    parser.add_argument(
        '--epoch', metavar='NUM', type=int, default=20,
        help='training epoch')
    parser.add_argument(
        '--batchsize', metavar='NUM', type=int, default=32,
        dest='batch_size', help='batch size')
    parser.add_argument(
        '--device', metavar='ID', type=int, default=-1, help='device id')
    parser.add_argument(
        '--seed', metavar='NUM', type=int, help='random seed')
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    if args.device >= 0:
        minn.set_device(minn.devices.CUDA(args.device))
    if args.seed is not None:
        np.random.seed(args.seed)
        minn.get_device().xp.random.seed(args.seed)
    train(args.train_file,
          args.valid_file,
          args.epoch,
          args.batch_size,
          args.learning_rate)
