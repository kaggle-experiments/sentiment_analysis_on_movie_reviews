from config import Config
from pprint import pprint, pformat
from pprint import pprint, pformat
import logging
log = logging.getLogger('main')
log.setLevel(logging.DEBUG)

from collections import namedtuple, defaultdict

from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize
"""
from tokenizer import word_tokenize
"""

VOCAB =  ['PAD', 'UNK', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""


RawSample = namedtuple('RawSample', ['id','sentence_id',  'sentence', 'sentiment'])
SentimentSample = namedtuple('Sample', ['id','sentence_id',  'tokens', 'sentiment'])

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
            
"""
heatmap
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('figure', figsize=(10, 5))

def heatmap(mat, filepath, xticks=[], yticks=[],  title='', xlabel='', ylabel=''):
    plt.matshow(mat, cmap='hot')
    plt.title(title)
    if xticks: plt.xticks(*xticks, rotation=90)
    if yticks: plt.yticks(*yticks, rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.savefig(filepath)
    plt.close()
    return 
