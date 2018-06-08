import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import numpy

import nli_data as data
from model_PRPN_nli import PRPN

parser = argparse.ArgumentParser(description='PennTreeBank PRPN Language Model')
parser.add_argument('--data', type=str, default='../datasets/multinli_1.0',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.003,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='dropout applied to output layers (0 = no dropout)')
parser.add_argument('--idropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--rdropout', type=float, default=0.5,
                    help='dropout applied to recurrent states (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--hard', action='store_true',
                    help='use hard sigmoid')
parser.add_argument('--res', type=int, default=0,
                    help='number of resnet block in predict network')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='./model/model_LM.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default=None,
                    help='path to save the final model')
parser.add_argument('--nslots', type=int, default=15,
                    help='number of memory slots')
parser.add_argument('--nlookback', type=int, default=5,
                    help='number of look back steps when predict gate')
parser.add_argument('--resolution', type=float, default=0.1,
                    help='syntactic distance resolution')
parser.add_argument('--device', type=int, default=0,
                    help='select GPU')
args = parser.parse_args()

torch.cuda.set_device(args.device)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
print("dict len: ", len(corpus.dictionary))

def batchify(data, bsz, random_start_idx=False, use_cuda=True):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    if random_start_idx:
        random.shuffle(data)
    data = data[0: nbatch * bsz]

    def list2batch(x_list):
        max_len = max([max(len(x['sen1_idx']), len(x['sen2_idx'])) for x in x_list])
        if max_len > 80: max_len = 80
        pre_b = torch.cuda.LongTensor(max_len, bsz).zero_()
        hyp_b = torch.cuda.LongTensor(max_len, bsz).zero_()
        labels = torch.cuda.LongTensor(bsz).zero_()

        for idx, x in enumerate(x_list):
            batch_len = max(len(x['sen1_idx']), len(x['sen2_idx']))
            if batch_len > max_len:
                batch_len = max_len
                x['sen1_idx'] = x['sen1_idx'][:max_len]
                x['sen2_idx'] = x['sen2_idx'][:max_len]
            pre_b[:len(x['sen1_idx']), idx] = x['sen1_idx']
            hyp_b[:len(x['sen2_idx']), idx] = x['sen2_idx']
            labels[idx] = x['gold_label']

        '''
        if use_cuda:
            pre_b.cuda()
            hyp_b.cuda()
            labels.cuda()
        '''
        return pre_b, hyp_b, labels

    
    data_batched = []
    for i in range(nbatch):
        batch = data[i * bsz: (i + 1) * bsz]
        pre, hyp, label = list2batch(batch)
        data_batched.append((pre, hyp, label))        
    return data_batched

eval_batch_size = 10
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = PRPN(ntokens, args.emsize, args.nhid, args.nlayers,
             args.nslots, args.nlookback, args.resolution,
             args.dropout, args.idropout, args.rdropout,
             args.tied, args.hard, args.res)

if not (args.load is None):
    with open(args.load, 'rb') as f:
        model = torch.load(f)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()


###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        if isinstance(h, list):
            return [repackage_hidden(v) for v in h]
        else:
            return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    pre, hyp, label = source[i]
    pre = Variable(pre, volatile=evaluation)
    hyp = Variable(hyp, volatile=evaluation)
    label = Variable(label)
    return pre, hyp, label


def evaluate(data_source):
    model.eval()
    total_loss = 0
    total_accu = 0
    #for i in range(0, data_source.size(0)-1, args.bptt):
    for i in range(len(data_source)):
        pre, hyp, label = get_batch(data_source, i, evaluation=True)
        logits = model(pre, hyp, eval_batch_size)
        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float()
        loss = criterion(input=logits, target=label)
        total_loss += len(pre) * loss.data
        total_accu += accuracy.data
    return total_loss*1.0/len(data_source), total_accu*1.0/len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = total_accu = 0
    cur_accuracy = 0.
    cur_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    train_data = batchify(corpus.train, args.batch_size, random_start_idx=True)
    #label = Variable(torch.cuda.LongTensor([0]*64))
    #label.cuda()
    for batch in range(len(train_data)):
        pre_t, hyp_t, label = get_batch(train_data, batch)
        optimizer.zero_grad()
        logits = model(pre_t, hyp_t, args.batch_size)

        label_pred = logits.max(1)[1]
        accuracy = torch.eq(label, label_pred).float().mean()
        loss = criterion(input=logits, target=label)
        total_loss += loss.data
        total_accu += accuracy.data
        cur_accuracy = 0.95 * cur_accuracy + 0.05 * accuracy.data[0]
        if batch == 0:
            cur_accuracy = accuracy.data[0]
        cur_loss = 0.95 * cur_loss + 0.05 * loss.data[0]
        if batch == 0:
            cur_loss = loss.data[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
       
        
        if batch % args.log_interval == 0 and batch > 0:
            #cur_loss = total_loss[0] / args.log_interval
            #cur_accuracy = total_accu[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | accuracy {:8.2f}'.format(
                epoch, batch, len(train_data) // args.batch_size, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, cur_accuracy))
            total_loss = 0
            start_time = time.time()
    

# Loop over epochs.
lr = args.lr
best_val_loss = None
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss, val_accuracy = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:5.2f}'.format(epoch, (time.time() - epoch_start_time),
              val_loss[0], val_accuracy[0]))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if best_val_loss is None or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        scheduler.step(val_loss[0])

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
