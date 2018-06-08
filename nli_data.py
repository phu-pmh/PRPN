import os
import torch
import random
import re
import cPickle
import numpy
from nltk import word_tokenize

random.seed(1111)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if self.word2idx.has_key(item):
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.iteritems():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print 'Number of words:', len(self.idx2word)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        dict_file_name = 'data/mnli_dict.pkl'
        train_path = os.path.join(path, 'multinli_1.0_train.jsonl')
        test_path = os.path.join(path,'multinli_1.0_dev_matched.jsonl')
        if os.path.exists(dict_file_name):
            self.dictionary = cPickle.load(open(dict_file_name, 'rb'))
        else:
            self.dictionary = Dictionary()
            self.add_words(train_path)
            self.dictionary.rebuild_by_freq()
            cPickle.dump(self.dictionary, open(dict_file_name, 'wb'))
            
        self.train, self.valid = self.tokenize(train_path, split=True)
        self.test = self.tokenize(test_path)

    def filter_words(self, sent):
        words = []
        for w in word_tokenize(sent):
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            words.append(w)
        return words


    def add_words(self, path):
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                data = eval(line)
                words = self.filter_words(data['sentence1'])
                words = ['<s>'] + words + ['</s>']
                for word in words:
                    self.dictionary.add_word(word)
                words = self.filter_words(data['sentence2'])
                words = ['<s>'] + words + ['</s>']
                for word in words:
                    self.dictionary.add_word(word)
                

    def tokenize(self, path, split=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        examples = []
        label_ids = {'neutral':0, 'entailment':1, 'contradiction':2}
        with open(path, 'r') as f:
            for line in f:
                data = eval(line)

                if data['gold_label'] == '-': 
                    continue
                sen1_words = self.filter_words(data['sentence1'])
                sen2_words = self.filter_words(data['sentence2'])
                sen1_words = ['<s>'] + sen1_words + ['</s>'] 
                sen2_words = ['<s>'] + sen2_words + ['</s>']
                data['sen1_tokenized'] = sen1_words
                data['sen2_tokenized'] = sen2_words
                sen1_idx = []
                sen2_idx = []
                for word in sen1_words:
                    sen1_idx.append(self.dictionary[word])
                data['sen1_idx'] = torch.LongTensor(sen1_idx)
                for word in sen2_words:
                    sen2_idx.append(self.dictionary[word])
                data['sen2_idx'] = torch.LongTensor(sen2_idx)
                data['gold_label'] = label_ids[data['gold_label']]
                del data['annotator_labels']
                del data['genre']
                examples.append(data)               
        
        if split:
            random.shuffle(examples)
            return examples[:-10000], examples[-10000:]        
        return examples

           

