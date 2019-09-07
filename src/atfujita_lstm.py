import gc
import logging
import os
import time
import math
import random
import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import spacy
import gensim
from keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer

import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

print(os.listdir("../input"))

SEED = 19840916
BATCH_SIZE = 512
NUM_EPOCHS = 7
maxlen = 220


# Data preprocessing
TEXT_COL = 'comment_text'
CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'


def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger


def seed_torch(seed=19840916):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clean_tag(text):
    if '[math]' in text:
        text = re.sub('\[math\].*?math\]', '[formula]', text)
    if 'http' in text or 'www' in text:
        text = re.sub('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+', '[url]', text)
    return text


# Use fast text as vocabulary
def words(text):
    return re.findall(r'\w+', text.lower())


def P(word):
    "Probability of `word`."
    # use inverse of rank as proxy
    # returns 0 if the word isn't in the dictionary
    return - WORDS.get(word, 0)


def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)


def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or [word])


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


def singlify(word):
    return "".join([letter for i, letter in enumerate(word) if i == 0 or letter != word[i-1]])


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))


def build_matrix(word_dict, lemma_dict, path, max_features, embed_size=300):

    embedding_index = load_embeddings(path)

    if path == CRAWL_EMBEDDING_PATH:
        nb_words = min(max_features, len(word_dict)+1)
        embedding_matrix = np.zeros((nb_words, embed_size))
    elif path == GLOVE_EMBEDDING_PATH:
        nb_words = min(max_features, len(word_dict)+1)
        embedding_matrix = np.zeros((nb_words, embed_size))

    unknown_words = []

    for key, l in word_dict.items():
        word = key

        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.lower()
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = re.sub('[0-9]+', '', key)
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        if "trump" in word:
            embedding_matrix[word_dict[key]] = embedding_index["trump"]
            continue
        if "drump" in word:
            embedding_matrix[word_dict[key]] = embedding_index["trump"]
            continue
        if "trunp" in word:
            embedding_matrix[word_dict[key]] = embedding_index["trump"]
            continue
        word = key.upper()
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = key.capitalize()
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = ps.stem(key)
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lc.stem(key)
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = sb.stem(key)
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        word = lemma_dict[key]
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word_dict[key]] = embedding_vector
            continue
        if len(key) > 1:
            word = correction(key)
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word_dict[key]] = embedding_vector
                continue
        unknown_words.append(key)

    return embedding_matrix, unknown_words


# Model
class LockedDropout(nn.Module):
    """ LockedDropout applies the same dropout mask to every time step.

    **Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.

    Args:
        p (float): Probability of an element in the dropout mask to be zeroed.
    """

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (:class:`torch.FloatTensor` [batch size, sequence length, rnn hidden size]): Input to
                apply dropout too.
        """
        if not self.training or not self.p:
            return x
        x = x.clone()
        mask = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'p=' + str(self.p) + ')'


class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in',
                                 nonlinearity='leaky_relu')

    def forward(self, x):
        return self.linear(x)


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise, self).__init__()

        self.stddev = stddev

    def forward(self, x):
        noise = torch.empty_like(x)
        noise.normal_(0, self.stddev)

        return x + noise


class Swish(nn.Module):

    def __init__(self, inplace=False):
        super().__init__()

        self.inplace = True

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


class NeuralNet(nn.Module):
    def __init__(self, batch_size=None, output_size=1, hidden_size=128,
                 input_size=150, embedding_matrix=None):
        super(NeuralNet, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table
        """

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_length = embedding_matrix.shape[1]

        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), sparse=False)
        self.gn = GaussianNoise(0.15)
        self.lockeddropout = LockedDropout(p=0.2)
        self.lstm = nn.LSTM(embedding_matrix.shape[1],
                            hidden_size,
                            bidirectional=True,
                            batch_first=False)
        self.gru = nn.GRU(hidden_size * 2,
                          hidden_size,
                          bidirectional=True,
                          batch_first=False)
        self.input = nn.Linear(embedding_matrix.shape[1], input_size)
        self.W_s1 = nn.Linear(hidden_size*2+input_size, 150, bias=True)
        self.W_s2 = nn.Linear(150, 65, bias=True)
        self.lnorm = torch.nn.LayerNorm(hidden_size*2+input_size,
                                        eps=1e-05, elementwise_affine=True)
        self.swish = Swish()
        self.W2 = Linear(2*hidden_size+input_size, 100)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.lnorm2 = torch.nn.LayerNorm(100, eps=1e-05, elementwise_affine=True)
        self.dropout_2 = nn.Dropout(0.2)
        self.label = Linear(100, output_size)
        self.linear_aux_out = nn.Linear(100, 7)

    def attention_net(self, lstm_output):

        """
        Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
        encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
        the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
        connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
        pos & neg.
        Arguments
        ---------
        lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
        ---------
        Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
                  attention to different parts of the input sentence.
        Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                      attn_weight_matrix.size() = (batch_size, 30, num_seq)
        """
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))  # / self.temper
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_sentence, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        """
        """
        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
        of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of
        its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
        state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
        vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
        dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.
        """
        input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        if self.training:
            input = self.gn(input)
        input = self.lockeddropout(input)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.empty(2, len(input[1]),
                                       self.hidden_size).cuda())  # Initial hidden state of the LSTM
            h_0 = nn.init.xavier_uniform_(h_0, gain=5/3)
            c_0 = Variable(torch.empty(2, len(input[1]),
                                       self.hidden_size).cuda())  # Initial cell state of the LSTM
            c_0 = nn.init.xavier_uniform_(c_0, gain=5/3)
        else:
            h_0 = Variable(torch.empty(2, batch_size, self.hidden_size).cuda())
            h_0 = nn.init.xavier_uniform_(h_0, gain=5/3)
            c_0 = Variable(torch.empty(2, batch_size, self.hidden_size).cuda())
            c_0 = nn.init.xavier_uniform_(c_0, gain=5/3)

        output, (final_hidden_state, _) = self.lstm(input, (h_0, c_0))
        output, _ = self.gru(output, final_hidden_state)

        input = torch.tanh(self.input(input))

        final_encoding = torch.cat((output, input), 2)
        final_encoding = final_encoding.permute(1, 0, 2)
        attn_weight_matrix = self.attention_net(final_encoding)
        final_encoding = torch.matmul(attn_weight_matrix, final_encoding)
        final_encoding = self.lnorm(final_encoding)
        final_encoding = self.swish(final_encoding)
        y = self.W2(final_encoding)
        y = y.permute(0, 2, 1)
        y = self.maxpool(y)
        y = y.squeeze(2)
        y = self.lnorm2(y)
        if self.training:
            y = self.gn(y)
        y = self.dropout_2(y)
        logits = self.label(y)
        aux_result = self.linear_aux_out(y)
        out = torch.cat([logits, aux_result], 1)

        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size, torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

        return loss


# code inspired from: https://github.com/anandsaha/pytorch.cyclic.learning.rate/blob/master/cls.py
class CyclicLR(object):
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)

        return lrs


# Stolen from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        #elif score < self.best_score:
        elif score > self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def custom_loss(data, targets):
    ''' Define custom loss function for weighted BCE on 'target' column '''
    bce_loss_1 = nn.BCEWithLogitsLoss(
        weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:2], targets[:, 2:3])
    bce_loss_3 = nn.BCEWithLogitsLoss()(data[:, 2:3], targets[:, 3:4])
    bce_loss_4 = nn.BCEWithLogitsLoss()(data[:, 3:4], targets[:, 4:5])
    bce_loss_5 = nn.BCEWithLogitsLoss()(data[:, 4:5], targets[:, 5:6])
    bce_loss_6 = nn.BCEWithLogitsLoss()(data[:, 5:6], targets[:, 6:7])
    bce_loss_7 = nn.BCEWithLogitsLoss()(data[:, 6:7], targets[:, 7:8])
    bce_loss_8 = nn.BCEWithLogitsLoss()(data[:, 7:8], targets[:, 8:9])

    return bce_loss_1 + bce_loss_2 + bce_loss_3 + bce_loss_4 \
           + bce_loss_5 + bce_loss_6 + bce_loss_7 + bce_loss_8


# refrence: https://www.kaggle.com/dborkan/benchmark-kernel
class SubmetricsAUC(object):
    def __init__(self, valid_df, pred_y):
        self.identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian',
                                 'christian', 'jewish', 'muslim', 'black',
                                 'white', 'psychiatric_or_mental_illness']
        self.SUBGROUP_AUC = 'subgroup_auc'
        self.BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
        self.BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

        self.TOXICITY_COLUMN = 'target'

        self.valid_df = valid_df
        self.pred_y = pred_y
        self.model_name = 'pred'
        self.valid_df[self.model_name] = self.pred_y
        self.valid_df = self.convert_dataframe_to_bool(self.valid_df)

    def compute_auc(self):

        bias_metrics_df = self.compute_bias_metrics_for_model(self.identity_columns,
                                                              self.model_name,
                                                              self.TOXICITY_COLUMN).fillna(0)

        final_score = self.get_final_metric(bias_metrics_df,
                                            self.calculate_overall_auc())

        return final_score

    @staticmethod
    def power_mean(series, p):
        total = sum(np.power(series, p))
        return np.power(total / len(series), 1 / p)

    @staticmethod
    def calculate_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    @staticmethod
    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    def convert_dataframe_to_bool(self, df):
        bool_df = df.copy()
        for col in ['target'] + self.identity_columns:
            self.convert_to_bool(bool_df, col)
        return bool_df

    def compute_subgroup_auc(self, subgroup, label, model_name):
        subgroup_examples = self.valid_df[self.valid_df[subgroup]]
        return self.calculate_auc(subgroup_examples[label],
                                  subgroup_examples[model_name])

    def compute_bpsn_auc(self, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = self.valid_df[self.valid_df[subgroup] & ~self.valid_df[label]]
        non_subgroup_positive_examples = self.valid_df[~self.valid_df[subgroup] & self.valid_df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return self.calculate_auc(examples[label], examples[model_name])

    def compute_bnsp_auc(self, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = self.valid_df[self.valid_df[subgroup] & self.valid_df[label]]
        non_subgroup_negative_examples = self.valid_df[~self.valid_df[subgroup] & ~self.valid_df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return self.calculate_auc(examples[label], examples[model_name])

    def compute_bias_metrics_for_model(self,
                                       subgroups,
                                       model,
                                       label_col,
                                       include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""
        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(self.valid_df[self.valid_df[subgroup]])
            }

            record[self.SUBGROUP_AUC] = self.compute_subgroup_auc(subgroup,
                                                                  label_col,
                                                                  model)
            record[self.BPSN_AUC] = self.compute_bpsn_auc(subgroup,
                                                          label_col,
                                                          model)
            record[self.BNSP_AUC] = self.compute_bnsp_auc(subgroup,
                                                          label_col,
                                                          model)
            records.append(record)
        return pd.DataFrame(records).sort_values('subgroup_auc',
                                                 ascending=True)

    def calculate_overall_auc(self):
        true_labels = self.valid_df[self.TOXICITY_COLUMN]
        predicted_labels = self.valid_df[self.model_name]
        return roc_auc_score(true_labels, predicted_labels)

    def get_final_metric(self, bias_df, overall_auc, power=-5, weight=0.25):
        bias_score = np.average([
            self.power_mean(bias_df[self.SUBGROUP_AUC], power),
            self.power_mean(bias_df[self.BPSN_AUC], power),
            self.power_mean(bias_df[self.BNSP_AUC], power)
        ])
        return (weight * overall_auc) + ((1 - weight) * bias_score)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)


# Training
def trainer(X_train, y_train, #X_test,
            embedding_matrix, train_ori):

    logger.info('Prepare folds')

    splits = list(StratifiedKFold(n_splits=10,
                                  shuffle=True,
                                  random_state=SEED).split(X_train,
                                                           y_train[:, 0]))
    train_preds = np.zeros((len(X_train)))

    for i, (train_idx, valid_idx) in enumerate(splits):
        x_train_fold = torch.tensor(X_train[train_idx], dtype=torch.long) #.cuda()
        y_train_fold = torch.tensor(y_train[train_idx], dtype=torch.float32)  #.cuda()

        x_val_fold = torch.tensor(X_train[valid_idx], dtype=torch.long)  #.cuda()
        y_val_fold = torch.tensor(y_train[valid_idx], dtype=torch.float32)#.cuda()

        train_df = train_ori.iloc[valid_idx, :]

        model = NeuralNet(embedding_matrix=embedding_matrix)
        model.cuda()

        step_size = 1500
        base_lr, max_lr = 0.0005, 0.003

        optimizer = AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.98),
                          weight_decay=0.0001)

        scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                             step_size=step_size, mode='exp_range',
                             gamma=0.99994)

        train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

        train = MyDataset(train)
        valid = MyDataset(valid)

        train_loader = torch.utils.data.DataLoader(train,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=4)
        valid_loader = torch.utils.data.DataLoader(valid,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=4)

        early_stopping = EarlyStopping(patience=3, verbose=True)

        del train, valid

        print(f'Fold {i + 1}')
        logger.info('Run model')
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()

            model.train()
            avg_loss = 0.
            for j, (x_batch, y_batch, index) in enumerate(train_loader):

                optimizer.zero_grad()

                y_pred = model(x_batch.cuda(non_blocking=True))
                if scheduler:
                    scheduler.batch_step()
                loss = custom_loss(y_pred, y_batch.cuda(non_blocking=True))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

                avg_loss += loss.item() / len(train_loader)

            model.eval()
            valid_preds_fold = np.zeros((x_val_fold.size(0)))
            avg_val_loss = 0.
            for j, (x_batch, y_batch, index) in enumerate(valid_loader):
                y_pred = model(x_batch.cuda(non_blocking=True)).detach()
                avg_val_loss = custom_loss(
                    y_pred, y_batch.cuda(non_blocking=True)) / len(valid_loader)

                valid_preds_fold[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = \
                    sigmoid(y_pred.cpu().numpy())[:, 0]

            score = SubmetricsAUC(train_df, valid_preds_fold).compute_auc()

            elapsed_time = time.time() - start_time
            print(
                'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} '
                '\t SubmetricsAUC={:.4f} \t time={:.2f}s'.format(
                    epoch + 1, NUM_EPOCHS, avg_loss, avg_val_loss, score,
                    elapsed_time))

            #early_stopping(avg_val_loss, model)
            early_stopping(score, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('./weights/checkpoint.pt'))
        tmp_dict = model.state_dict()
        del tmp_dict['word_embeddings.weight']
        torch.save(tmp_dict, f'./weights/atfujita_lstm_{i + 1}.pt')

        valid_preds_fold = np.zeros((x_val_fold.size(0)))

        for j, (x_batch, y_batch, index) in enumerate(valid_loader):

            y_pred = model(x_batch.cuda(non_blocking=True)).detach()
            valid_preds_fold[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = \
                sigmoid(y_pred.cpu().numpy())[:, 0]

        score = SubmetricsAUC(train_df, valid_preds_fold).compute_auc()
        print('Fold Best Model CV: ', score)
        train_preds[valid_idx] = valid_preds_fold
        del model, train_loader, valid_loader, valid_preds_fold, y_pred, loss
        del x_train_fold, y_train_fold, x_val_fold, y_val_fold, train_df
        torch.cuda.empty_cache()

    logger.info('Complete run model')

    return train_preds


def submit(sub_preds):
    logger.info('Prepare submission')
    submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv',
                             index_col='id')
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)


logger = get_logger()

ps = PorterStemmer()
lc = LancasterStemmer()
sb = SnowballStemmer("english")

# https://www.kaggle.com/cpmpml/spell-checker-using-word2vec
spell_model = gensim.models.KeyedVectors.load_word2vec_format(CRAWL_EMBEDDING_PATH)
words = spell_model.index2word
w_rank = {}
for k, word in enumerate(words):
    w_rank[word] = k
WORDS = w_rank

del spell_model, words, w_rank
gc.collect()

with open('WORDS.pickle', 'wb') as f:
    pickle.dump(WORDS, f)


def main():
    seed_torch(SEED)
    train = pd.read_csv(
        '../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv',
        index_col='id')

    train_ori = train.copy()
    train_ori = train_ori.reset_index(drop=True)

    y_aux_train = train[['target', 'severe_toxicity', 'obscene',
                         'identity_attack', 'insult',
                         'threat',
                         'sexual_explicit'
                         ]]
    y_aux_train = y_aux_train.fillna(0)

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    # Overall
    weights = np.ones((len(train),)) / 4
    # Subgroup
    weights += (train[identity_columns].fillna(0).values >= 0.5).sum(
        axis=1).astype(bool).astype(np.int) / 4
    # Background Positive, Subgroup Negative
    weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values < 0.5).sum(
                     axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    # Background Negative, Subgroup Positive
    weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                 (train[identity_columns].fillna(0).values >= 0.5).sum(
                     axis=1).astype(bool).astype(np.int)) > 1).astype(
        bool).astype(np.int) / 4
    loss_weight = 1.0 / weights.mean()
    print(loss_weight)

    y_train = np.vstack(
        [(train['target'].values >= 0.5).astype(np.int), weights]).T

    y_train = np.hstack([y_train, y_aux_train])
    print(y_train.shape)

    # fill up the missing values
    train = train['comment_text'].fillna(' ').values

    max_features = None
    print("Spacy NLP ...")
    start_time = time.time()
    num_train_data = y_train.shape[0]
    text_list = list(train)

    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner', 'tagger'])
    nlp.vocab.add_flag(
        lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS,
        spacy.attrs.IS_STOP)

    with open('nlp.pickle', 'wb') as f:
        pickle.dump(nlp, f)

    word_dict = {}
    word_index = 1
    lemma_dict = {}
    docs = nlp.pipe(text_list, n_threads=6)
    word_sequences = []

    for doc in tqdm(docs):
        word_seq = []
        for token in doc:
            if (token.text not in word_dict) and (token.pos_ is not "PUNCT"):
                word_dict[token.text] = word_index
                word_index += 1
                lemma_dict[token.text] = token.lemma_
            if token.pos_ is not "PUNCT":
                word_seq.append(word_dict[token.text])
        word_sequences.append(word_seq)

    del docs, train, text_list#, test
    gc.collect()

    with open('word_dict.pickle', 'wb') as f:
        pickle.dump(word_dict, f)
    f = open('word_dict.txt', 'w')  # 書き込みモードで開く
    for key, value in sorted(word_dict.items()):
        f.write(f'{key} {value}\n')
    f.close()

    X_train = word_sequences[:num_train_data]
    print("--- %s seconds ---" % (time.time() - start_time))

    del word_sequences, nlp
    gc.collect()

    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')

    train_idx = np.random.permutation(len(X_train))
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
    train_ori = train_ori.ix[train_idx, :]

    max_features = max_features or len(word_dict)+1
    print('max_feature: ', max_features)

    crawl_matrix, unknown_words_crawl = build_matrix(word_dict, lemma_dict,
                                                     CRAWL_EMBEDDING_PATH,
                                                     max_features)
    print('n unknown words (crawl): ', len(unknown_words_crawl))
    with open("unknown_words_crawl.txt", 'wt') as f:
        for ele in unknown_words_crawl:
            f.write(ele + '\n')
    f.close()

    glove_matrix, unknown_words_glove = build_matrix(word_dict, lemma_dict,
                                                     GLOVE_EMBEDDING_PATH,
                                                     max_features)
    print('n unknown words (glove): ', len(unknown_words_glove))
    with open("unknown_words_glove.txt", 'wt') as f:
        for ele in unknown_words_glove:
            f.write(ele + '\n')
    f.close()

    del word_dict, lemma_dict
    del unknown_words_crawl, unknown_words_glove
    gc.collect()

    embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)

    del crawl_matrix
    del glove_matrix
    gc.collect()

    with open('embedding_matrix.pickle', 'wb') as f:
        pickle.dump(embedding_matrix, f)

    print('embedding shape: ', embedding_matrix.shape)

    train_preds = trainer(X_train, y_train,
                          embedding_matrix,
                          train_ori
                          )

    del embedding_matrix

    print(SubmetricsAUC(train_ori, train_preds).compute_auc())
    y_train = y_train[:, 0]
    y_train = y_train[:, np.newaxis]
    print(roc_auc_score(y_train > 0.5, train_preds))


if __name__ == '__main__':
    main()
