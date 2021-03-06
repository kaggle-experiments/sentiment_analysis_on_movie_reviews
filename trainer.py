import logging
from config import Config
from pprint import pprint, pformat

from pprint import pprint, pformat
import logging
log = logging.getLogger('main')
log.setLevel(logging.DEBUG)

#log.setLevel(logging.DEBUG)

from debug import memory_consumed
from utilz import ListTable
from tqdm import tqdm as _tqdm

import torch

from torch import optim, nn

from collections import namedtuple

Feeder = namedtuple('Feeder', ['train', 'test'])

def tqdm(a):
    return _tqdm(a) if Config().tqdm else a

class FLAGS:
    CONTINUE_TRAINING = 0
    STOP_TRAINING = 1

class Runner(object):
    def __init__(self, model, *args, **kwargs):
        self._model = model

    def run(self, input):
        model_output = self.model(*input)
        return model_output

    @property
    def model(self):
        return self._model
    
class Averager(list):
    def __init__(self, filename=None, *args, **kwargs):
        super(Averager, self).__init__(*args, **kwargs)
        if filename:
            open(filename, 'w').close()

        self.filename = filename
        
    @property
    def avg(self):
        if len(self):
            return sum(self)/len(self)
        else:
            return 0


    def __str__(self):
        if len(self) > 0:
            return 'min/max/avg/latest: {:0.5f}/{:0.5f}/{:0.5f}/{:0.5f}'.format(min(self), max(self), self.avg, self[-1])
        
        return '<empty>'

    def append(self, a):
        try:
            super(Averager, self).append(a.data[0])
        except:
            super(Averager, self).append(a)
            
    def empty(self):
        del self[:]

    def write_to_file(self):
        if self.filename:
            with open(self.filename, 'a') as f:
                f.write(self.__str__() + '\n')

class EpochAverager(Averager):
    def __init__(self, filename=None, *args, **kwargs):
        super(EpochAverager, self).__init__(*args, **kwargs)
        self.epoch_cache = Averager(filename, *args, *kwargs)

    def cache(self, a):
        self.epoch_cache.append(a)

    def clear_cache(self):
        super(EpochAverager, self).append(self.epoch_cache.avg)
        self.epoch_cache.empty()
        
class Trainer(object):
    def __init__(self, name, runner=None, model=None,
                 feeder = None,
                 optimizer=None,
                 loss_function = None,
                 accuracy_function=None,
                 f1score_function=None,
                 epochs=10000, checkpoint=1,
                 directory='results',
                 *args, **kwargs):

        self.name  = name
        self.__build_model_group(runner, model, *args, **kwargs)
        self.__build_feeder(feeder, *args, **kwargs)

        self.epochs     = epochs
        self.checkpoint = checkpoint

        self.optimizer     = optimizer     if optimizer     else optim.SGD(self.runner.model.parameters(), lr=0.01, momentum=0.1)

        # necessary metrics
        self.train_loss = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, name, 'metrics',  'train_loss'))
        self.test_loss  = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, name, 'metrics', 'test_loss'))
        self.accuracy_function = accuracy_function if accuracy_function else self._default_accuracy_function

        self.accuracy   = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, name, 'metrics', 'accuracy'))
        self.loss_function = loss_function if loss_function else nn.NLLLoss()

        # optional metrics
        self.f1score_function = f1score_function
        self.precision = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, name, 'metrics', 'precision'))
        self.recall = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, name, 'metrics', 'recall'))
        self.f1score   = EpochAverager(filename = '{}/{}/{}.{}'.format(directory, name, 'metrics', 'f1score'))

        self.metrics = [self.train_loss, self.test_loss, self.accuracy, self.precision, self.recall, self.f1score]
        
        self.best_model = (0,  self.runner.model.state_dict())
        

    def __build_model_group(self, runner, model, *args, **kwargs):
        assert model is not None or runner is not None, 'both model and runner are None, fatal error'
        if runner:
            self.runner = runner
        else:
            self.runner = Runner(model)
            
    def __build_feeder(self, feeder, *args, **kwargs):
        assert feeder is not None, 'feeder is None, fatal error'
        self.feeder = feeder

    def save_best_model(self):
        log.info('saving the last best model... to {} with accuracy: {}'.format( '{}.{}'.format(self.name, 'pth'), self.best_model[0]))
        torch.save(self.best_model[1], '{}.{}'.format(self.name, 'pth'))
        
    def train(self, test_drive=False):
        self.runner.model.train()
        for epoch in range(self.epochs):
            log.critical('memory consumed : {}'.format(memory_consumed()))            

            if self.do_every_checkpoint(epoch) == FLAGS.STOP_TRAINING:
                log.info('loss trend suggests to stop training')
                self.save_best_model()
                return
            
            for j in tqdm(range(self.feeder.train.num_batch)):
                self.optimizer.zero_grad()
                _, i, t = self.feeder.train.next_batch()
                output = self.runner.run(i)
                loss = self.loss_function( output, t )
                self.train_loss.append(loss)

                loss.backward()
                self.optimizer.step()
                
                if test_drive and j >= test_drive:
                    log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))
                    return
            
            log.info('-- {} -- loss: {}'.format(epoch, self.train_loss))            
            
            
            for m in self.metrics:
                m.write_to_file()
                
        self.runner.model.eval()
        return True
        
    def do_every_checkpoint(self, epoch, early_stopping=True):
        if epoch % self.checkpoint != 0:
            return
        self.runner.model.eval()
        for j in tqdm(range(self.feeder.test.num_batch)):
            _, i, t = self.feeder.train.next_batch()
            output =  self.runner.run(i)

            loss = self.loss_function(output, t)
            self.test_loss.cache(loss)
            accuracy = self.accuracy_function(output, t)
            self.accuracy.cache(accuracy)

            if self.f1score_function:
                precision, recall, f1score = self.f1score_function(output, t)
                self.precision.append(precision)
                self.recall.append(recall)
                self.f1score.append(f1score)

                
        log.info('-- {} -- loss: {}, accuracy: {}'.format(epoch, self.test_loss.epoch_cache, self.accuracy.epoch_cache))
        if self.f1score_function:
            log.info('-- {} -- precision: {}'.format(epoch, self.precision))
            log.info('-- {} -- recall: {}'.format(epoch, self.recall))
            log.info('-- {} -- f1score: {}'.format(epoch, self.f1score))

        self.test_loss.clear_cache()
        self.accuracy.clear_cache()
        
        log.info('{} < {} ?'.format(self.best_model[0], self.accuracy[-1]))
        if self.best_model[0] <= self.accuracy[-1]:
            log.info('Yes. best model bested the best model')
            self.best_model = (self.accuracy[-1], self.runner.model.state_dict())
        
        if early_stopping:
            return self.loss_trend()

    def loss_trend(self, falloff = 15):
        if len(self.test_loss) > falloff:
            losses = self.test_loss[-falloff:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > falloff-1:
                return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING


    def _default_accuracy_function(self):
        return -1

    
class Predictor(object):
    def __init__(self, runner=None, model=None,
                 feed = None,
                 repr_function = None,
                 *args, **kwargs):
        
        self.__build_model_group(runner, model, *args, **kwargs)
        self.__build_feed(feed, *args, **kwargs)
        self.repr_function = repr_function
        
    def __build_model_group(self, runner, model, *args, **kwargs):
        assert model is not None or runner is not None, 'both model and runner are None, fatal error'
        if runner:
            self.runner = runner
        else:
            self.runner = Runner(model)
            
    def __build_feed(self, feed, *args, **kwargs):
        assert feed is not None, 'feed is None, fatal error'
        self.feed = feed
        
    def predict(self,  batch_index=0):
        log.debug('batch_index: {}'.format(batch_index))
        _, i, *__ = self.feed.nth_batch(batch_index)
        log.debug('input_shape: {}'.format([j.shape for j in i]))
        self.runner.model.eval()
        output = self.runner.run(i)
        results = ListTable()
        results.extend( self.repr_function(output, self.feed, batch_index) )
        output_ = output
        return output_, results
