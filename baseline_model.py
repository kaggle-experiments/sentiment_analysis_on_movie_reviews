import os
import json
import time
import random
from pprint import pprint, pformat
import logging
log = logging.getLogger('main')
log.setLevel(logging.INFO)

from config import Config

from logger import model_logger
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from trainer import Trainer, Feeder, Predictor
from datafeed import DataFeed, MultiplexedDataFeed
from utilz import tqdm, ListTable

from functools import partial

from collections import namedtuple, defaultdict
import itertools

from utilz import logger
from utilz import PAD, pad_seq, word_tokenize
from utilz import VOCAB
from utilz import RawSample, SentimentSample as Sample

import numpy as np
from tokenstring import TokenString

SELF_NAME = os.path.basename(__file__).replace('.py', '')

def build_sentimentnet_sample(raw_sample):
    sentence = TokenString(raw_sample.sentence.strip(' \n\t'), word_tokenize)
    return Sample(raw_sample.id, raw_sample.sentence_id.lower(),
                  sentence,
                  raw_sample.sentiment.strip(' \n\t'))

def prep_samples_for_sentimentnet(dataset):
    ret = []
    vocabulary = defaultdict(int)
    labels = defaultdict(int)

    dataset = {s.id:s for s in dataset}    
    for i, (sid, sample) in tqdm(enumerate(dataset.items())):
        try:
            sample = build_sentimentnet_sample(sample)
            for token in sample.tokens:
                vocabulary[token] += 1
            labels[sample.sentiment] += 1
            ret.append(sample)
        except KeyboardInterrupt:
            return
        except:
            log.exception('at id: {}, sid: {}'.format(i, sid))

    return ret, vocabulary, labels


# ## Loss and accuracy function
def loss(output, target, loss_function=nn.NLLLoss(), scale=1, *args, **kwargs):
    target = Variable(torch.LongTensor(target[0]), requires_grad=False)
    if Config().cuda: target = target.cuda()
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))

    return loss_function(output, target)

def accuracy(output, target, *args, **kwargs):
    output = output.max(-1)[1]
    target = Variable(torch.LongTensor(target[0]))
    if Config().cuda: target = target.cuda()
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))
    return (output == target).sum().float()/output.size(0)

def repr_function(output, feed, batch_index, VOCAB, LABELS, raw_samples):
    results = []
    output = output.data.max(dim=-1)[1].cpu().numpy()
    indices, (seq, ), (labels,) = feed.nth_batch(batch_index)
    for idx, op, se, la in zip(indices, output, seq, labels):
        #sample = list(filter(lambda x: x.id == int(str(idx).split('.')[0]), raw_samples))[0]
        results.append([ ' '.join(feed.data_dict[idx].tokens), feed.data_dict[idx].sentiment, LABELS[op] ])

    return results

def batchop(datapoints, WORD2INDEX, LABEL2INDEX, *args, **kwargs):
    indices = [d.id for d in datapoints]
    seq = []
    label = []
    for d in datapoints:
        seq.append([WORD2INDEX[w] for w in d.tokens])
        label.append(LABEL2INDEX[d.sentiment])

    seq = pad_seq(seq)
    
    batch =  indices, (np.array(seq),), (np.array(label),)
    return batch

def experiment(VOCAB, LABELS, WORD2INDEX, LABEL2INDEX, raw_samples, datapoints=[[], []], eons=1000, epochs=10, checkpoint=1):
    try:
        try:
            model =  BiLSTMDecoderModel(Config(), len(VOCAB),  len(LABELS))
            if Config().cuda:  model = model.cuda()
            model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, '.pth')))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')
            model =  BiLSTMDecoderModel(Config(), len(VOCAB),  len(LABELS))
            if Config().cuda:  model = model.cuda()
        print('**** the model', model)

        name = SELF_NAME
        _batchop = partial(batchop, LABEL2INDEX=LABEL2INDEX)
        train_feed     = DataFeed(name, datapoints[0], batchop=_batchop, vocab=WORD2INDEX, batch_size=256)
        test_feed      = DataFeed(name, datapoints[1], batchop=_batchop, vocab=WORD2INDEX, batch_size=256)
        predictor_feed = DataFeed(name, datapoints[1], batchop=_batchop, vocab=WORD2INDEX, batch_size=128)

        loss_weight=Variable(torch.Tensor([0.1, 1, 1]))
        if Config.cuda: loss_weight = loss_weight.cuda()
        _loss = partial(loss, loss_function=nn.NLLLoss())
        trainer = Trainer(name=name,
                          model=model, 
                          loss_function=_loss, accuracy_function=accuracy, 
                          checkpoint=checkpoint, epochs=epochs,
                          feeder = Feeder(train_feed, test_feed))

        predictor = Predictor(model=model, feed=predictor_feed, repr_function=partial(repr_function, VOCAB=VOCAB, LABELS=LABELS, raw_samples=raw_samples))
        
        for e in range(eons):
            dump = open('results/experiment_attn.csv', 'a')
            dump.write('#========================after eon: {}\n'.format(e))
            dump.close()
            log.info('on {}th eon'.format(e))

            with open('results/experiment_attn.csv', 'a') as dump:
                results = ListTable()
                for ri in range(predictor_feed.num_batch):
                    output, _results = predictor.predict(ri)
                    results.extend(_results)
                dump.write(repr(results))
            if not trainer.train():
                raise Exception
        
    except :
        log.exception('####################')
        trainer.save_best_model()

        return locals()

class BiLSTMDecoderModel(nn.Module):
    def __init__(self, Config, input_vocab_size, output_vocab_size):
        super(BiLSTMDecoderModel, self).__init__()
        self.input_vocab_size = input_vocab_size
        
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = Config.hidden_dim
        self.embed_dim = Config.embed_dim

        self.embed = nn.Embedding(self.input_vocab_size, self.embed_dim)

        self.fencode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.bencode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(0.01)
        
        self.classify = nn.Linear(2*self.hidden_dim, self.output_vocab_size)
        
        self.log = model_logger.getLogger('model')
        self.size_log = self.log.getLogger('size')
        self.log.setLevel(logging.DEBUG)
        self.size_log.setLevel(logging.INFO)
            
        if Config.cuda:
            self.cuda()

    def cpu(self):
        super(BiLSTMDecoderModel, self).cpu()
        return self
    
    def cuda(self):
        super(BiLSTMDecoderModel, self).cuda()
        return self
    
    def __(self, tensor, name=''):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.size_log.debug('{}[{}] -> {}'.format(name, i, tensor[i].size()))

        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
                
        return tensor
        
    def init_hidden(self, batch_size):
        ret = torch.zeros(batch_size, self.hidden_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def init_label(self, batch_size):
        ret = torch.zeros(batch_size, self.embed_dim)
        if Config().cuda: ret = ret.cuda()
        return Variable(ret)
    
    def forward(self, seq):
        seq      = self.__( Variable(torch.LongTensor(seq)), 'seq')

        if seq.dim() == 1: seq = seq.unsqueeze(0)
        
        dropout  = self.dropout
        
        if not self.training:
            dropout = lambda i: i
        
        if Config().cuda: 
            seq = seq.cuda()
            
        batch_size, seq_size = seq.size()
        pad_mask = (seq > 0).float()
        seq_emb = self.__(   dropout( F.tanh(self.embed(seq)) ), 'seq_emb'   )

        seq_emb = seq_emb.transpose(1, 0)

        foutputs, boutputs = [], []
        foutput = self.init_hidden(batch_size), self.init_hidden(batch_size)
        boutput = self.init_hidden(batch_size), self.init_hidden(batch_size)
        for i in range(seq_size):
            foutput = self.__(  self.fencode(seq_emb[ i], foutput), 'foutput'   )
            boutput = self.__(  self.bencode(seq_emb[-i], boutput), 'boutput'   )
            foutputs.append(foutput[0])
            boutputs.append(boutput[0])

        boutputs = list(reversed(boutputs))
        foutputs, boutputs = torch.stack(foutputs), torch.stack(boutputs)
        seq_repr = self.__(  torch.cat([foutputs[-1], boutputs[-1]], dim=-1), 'seq_repr'  )

        logits = self.__(  self.classify(seq_repr), 'logits'  )
        return F.log_softmax(logits, dim=-1)
        
    
import sys
if __name__ == '__main__':
    dataset = open('train.tsv', 'r').readlines()[1:]
    for i,s in enumerate(dataset):
        dataset[i] = RawSample(*s.split('\t'))

    print('raw dataset size: {}'.format(len(dataset)))
    labelled_samples, vocabulary, labels = prep_samples_for_sentimentnet(dataset)
    #log.setLevel(logging.DEBUG)
    log.debug(pformat(labelled_samples[:10]))
    log.debug(pformat(vocabulary))
    log.setLevel(logging.INFO)
    
    VOCAB += list(vocabulary.keys())
    LABELS = sorted(list(labels.keys()))
    WORD2INDEX = { w:i for i,w in enumerate(VOCAB) }
    LABEL2INDEX = { w:i for i,w in enumerate(LABELS) }

    
    if sys.argv[1] == 'train':
        pivot = int( Config().split_ratio * len(labelled_samples) )
        random.shuffle(labelled_samples)
        train_set, test_set = labelled_samples[:pivot], labelled_samples[pivot:]
        
        train_set = sorted(train_set, key=lambda x: len(x.tokens))
        test_set = sorted(test_set, key=lambda x: len(x.tokens))
        
        experiment(VOCAB, LABELS, WORD2INDEX, LABEL2INDEX, dataset, datapoints=[train_set, test_set])
        
    if sys.argv[1] == 'predict':
        model =  BiLSTMDecoderModel(Config(), len(VOCAB),  len(LABELS))
        if Config().cuda:  model = model.cuda()
        model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, '.pth')))
        start_time = time.time()
        strings = sys.argv[2]
        
        s = [WORD2INDEX[i] for i in word_tokenize(strings)] + [WORD2INDEX['PAD']]
        e1, e2 = [WORD2INDEX['ENTITY1']], [WORD2INDEX['ENTITY2']]
        output = model(s, e1, e2)
        output = output.data.max(dim=-1)[1].cpu().numpy()
        label = LABELS[output[0]]
        print(label)

        duration = time.time() - start_time
        print(duration)
