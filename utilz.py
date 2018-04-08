from config import Config
from pprint import pprint, pformat
from logger import utilz_logger
log = utilz_logger.getLogger('main')
import logging
log.setLevel(logging.INFO)

from collections import namedtuple, defaultdict

from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize



RLABELS = ['NEUTRAL', 'ADVERSE', 'INDICATION']
ELABELS = ['O', 'B-Disease', 'B-Drug', 'I-Disease', 'I-Drug', 'L-Disease', 'L-Drug', 'U-Disease', 'U-Drug']
VOCAB =  ['PAD', 'UNK', 'EOS']

PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""


RawSample = namedtuple('RawSample', ['id','sentence',  'diseases', 'drugs', 'ae'])
EntitySample = namedtuple('Sample', ['id','tokens', 'labels'])
RelationSample = namedtuple('Sample', ['id', 'tokens', 'entity1', 'entity2', 'relation'])


"""
Logging utils
"""
def logger(func, dlevel=logging.INFO):
    def wrapper(*args, **kwargs):
        level = log.getEffectiveLevel()
        log.setLevel(level)
        ret = func(*args, **kwargs)
        log.setLevel(level)
        return ret
    
    return wrapper


from pprint import pprint, pformat
from tqdm import tqdm as _tqdm
from config import Config

def tqdm(a):
    return _tqdm(a) if Config().tqdm else a


def squeeze(lol):
    """
    List of lists to List

    Args:
        lol : List of lists

    Returns:
       List 

    """
    return [ i for l in lol for i in l ]

"""
    util functions to enable pretty print on namedtuple

"""
def _namedtuple_repr_(self):
    return pformat(self.___asdict())

def ___asdict(self):
    d = self._asdict()
    for k, v in d.items():
        if hasattr(v, '_asdict'):
            d[k] = ___asdict(v)

    return dict(d)


"""
# Batching utils   
"""
import numpy as np
def seq_maxlen(seqs):
    return max([len(seq) for seq in seqs])

def pad_seq(seqs, maxlen=0, PAD=PAD):
    def pad_seq_(seq):
        return seq[:maxlen] + [PAD]*(maxlen-len(seq))

    if len(seqs) == 0:
        return seqs
    
    if type(seqs[0]) == type([]):
        maxlen = maxlen if maxlen else seq_maxlen(seqs)
        seqs = [ pad_seq_(seq) for seq in seqs ]
    else:
        seqs = pad_seq_(seqs)
        
    return seqs


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
    the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
    IPython Notebook. 
    Taken from http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/"""
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

    def __repr__(self):
        lines = []
        for i in self:
            lines.append('|'.join(i))
        log.debug('number of lines: {}'.format(len(lines)))
        return '\n'.join(lines + ['\n'])
            
def lcm(x, y):
   lcm = x if x > y else y 
   while not (lcm % x == 0 and lcm % y == 0):  lcm += 1
   return lcm


"""
Dataset Utils
"""

def tokenized_indices(tokens, subspan, condition=lambda x, y: x == y, start_at=0):
    log.debug(pformat(tokens))
    log.debug(pformat(subspan))
    subspan_len = len(subspan)
    tokens_len = len(tokens)
    _subspan_len = 0
    index = start_at
    log.debug(locals())
    try:
        while index < tokens_len and not _subspan_len == subspan_len:
            _subspan_len = 0
            log.debug('index,_subspan_len: {}, {}'.format(index, _subspan_len))
            log.debug('{} ==> {} '.format(tokens[index + _subspan_len], subspan[_subspan_len]))
            while (_subspan_len < subspan_len
                   and index < len(tokens)
                   and condition(tokens[index + _subspan_len], subspan[_subspan_len])):
                log.debug('index: {} _subspan: {} '.format(index, _subspan_len))
                log.debug('  {} ==> {} '.format(tokens[index + _subspan_len], subspan[_subspan_len]))
                _subspan_len += 1

            index += 1

        index -= 1
        if index < len(tokens) - subspan_len:
            log.debug('returing ({}, {})'.format(index, index + _subspan_len))
            return index, index + _subspan_len
    except KeyboardInterrupt:
        return
    except:
        log.exception(pformat(locals()))
        return

def dataset_statistics(dataset):
    samples = defaultdict(defaultdict)
    seq_length_histogram = defaultdict(int)

def single_token_samples(dataset):
    samples = []
    for sample in dataset:
        if all( [len(word_tokenize(d)) == 1 for d in sample.drugs.split(',')]  )  and all( [len(word_tokenize(d)) == 1 for d in sample.diseases.split(',')]):
                 samples.append(sample)

    return samples


# dataset is list of Sample objects
@logger
def build_drug_dictionary(dataset):
    log.info('build_drug_dictinary')
    drugs = defaultdict(int)
    for i, sample in enumerate(dataset):
        for drug in sample.drugs.split(','):
            drugs[drug] += 1
        
    return dict(drugs)

@logger
def build_disease_dictionary(dataset):
    diseases = defaultdict(int)
    samples = [s for s in dataset if ',' in s.diseases]
    for i, sample in enumerate(samples):
        for disease in sample.diseases.split(','):
            diseases[disease] += 1
               
    return dict(diseases)
        
def build_drug_disease_mapping(dataset):
    pass
