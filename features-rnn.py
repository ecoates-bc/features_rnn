import csv, re
import random

import torch
import torch.nn as nn

EMBEDDING_DIM = 50
RNN_HIDDEN_DIM = 50
RNN_LAYERS = 1
MAXWFLEN = 40
EPOCHS = 10


def accuracy(sys,gold):
    assert len(sys) == len(gold)
    total_correct = 0
    for x, y in zip(sys, gold):
        if x == y:
            total_correct += 1
    return (total_correct / len(gold)) * 100


class FeatureEncoder:
    """
    An encoder that takes in an IPA symbol and returns a vector of phonological features + room for tag
    """
    def __init__(self):
        self.feature_dict = self.read_ipa2hayes('data/ipa2hayes.csv')
        self.alphabet = self.feature_dict.keys()
        self.tag_encoder = TagEncoder()
        self.num_features = len(self.feature_dict['a'][0]) + self.tag_encoder.num_tags
        self.dim = len(self.alphabet) + self.tag_encoder.num_tags

    def read_ipa2hayes(self, path):
        # create a dict of IPA symbols and feature vectors
        feature_dict = {}
        with open(path, 'r') as ipa2hayes:
            csvreader = csv.reader(ipa2hayes, delimiter=',')
            next(csvreader, None)  # skip header
            i = 0
            for line in csvreader:
                vec = ([self.convert_elem(e) for e in line[1:]], i)
                feature_dict[line[0]] = vec
                i+=1
        return feature_dict

    def get_features(self, char):
        try:
            return self.feature_dict[char][0] + self.tag_encoder.empty_vec()
        except KeyError:
            return [0 for i in range(self.num_features)]

    def convert_elem(self, elem):
        # convert +, - to 2, 1
        if elem == '+':
            return 1
        elif elem == '-':
            return -1
        else:
            return 0

    def encode(self, char):
        try:
            n = self.feature_dict[char][1]
            return n
        except KeyError:
            n = self.tag_encoder.get_idx(char)
            return n + len(self.feature_dict) - self.tag_encoder.num_tags

    def decode(self, n):
        for key in self.feature_dict:
            if self.feature_dict[key][1] == n:
                return key


class OneHotEncoder:
    """
    An encoder that takes in a character and returns a one-hot vector + room for tag
    """
    def __init__(self):
        self.char_dict = self.read_dataset('data/fra_verbs_ORTH.tsv')
        self.num_chars = len(self.char_dict)
        self.tag_encoder = TagEncoder()
        self.alphabet = list(self.char_dict.keys()) + list(self.tag_encoder.tag_dict.keys())
        self.dim = self.num_chars + self.tag_encoder.num_tags

    def read_dataset(self, path):
        # create a list of indexes for the alphabet
        idx_dict = {}
        i = 0
        dset = [line for line in open(path, 'r').readlines()]
        for line in dset:
            line = re.sub(r'(\t[^\t]*$)|\t', '', line)
            for char in line:
                if char not in idx_dict:
                    idx_dict[char] = i
                    i += 1
        return idx_dict

    def encode(self, char):
        try:
            n = self.char_dict[char]
            return n
        except KeyError:
            n = self.tag_encoder.get_idx(char)
            return n + self.dim - self.tag_encoder.num_tags

    def decode(self, n):
        return self.alphabet[n]


class TagEncoder:
    """
    Returns a mini-one-hot vector for tags, for use in the other encoders
    """
    def __init__(self):
        self.tag_dict = self.read_tags('data/fra_verbs_IPA.tsv')
        self.tag_dict['<EOS>'] = len(self.tag_dict)
        self.tag_dict['<SOS>'] = len(self.tag_dict)
        self.num_tags = len(self.tag_dict)

    def read_tags(self, path):
        tag_dict = {}
        dset = [line for line in open(path, 'r').readlines()]
        i = 0
        for line in dset:
            tag = re.sub(r'(^([^\t]*\t){2})|\n', '', line)
            if tag not in tag_dict:
                tag_dict[tag] = i
                i += 1
        return tag_dict

    def encode(self, tag):
        vect = self.empty_vec()
        try:
            vect[self.tag_dict[tag]] = 1
            return vect
        except IndexError:
            return vect

    def empty_vec(self):
        return [0 for i in range(self.num_tags)]

    def get_idx(self, tag):
        return self.tag_dict[tag]


NUM_FEATURES = FeatureEncoder().num_features


class Dataset:
    """
    Read in the IPA/orthography datasets, create aligned entries (ipa, orth)
    Split into train/dev/test, shuffle data
    """
    def __init__(self, dev_ratio, test_ratio):
        self.ipa = [self.format(line) for line in open('data/fra_verbs_IPA.tsv', 'r').readlines()]
        self.orth = [self.format(line) for line in open('data/fra_verbs_ORTH.tsv', 'r').readlines()]
        self.dataset = [ElemTuple(WordLine(self.ipa[i]), WordLine(self.orth[i])) for i in range(len(self.ipa))]
        self.dataset = self.split_into_test_train(self.dataset, dev_ratio, test_ratio)

    def format(self, line):
        split = line.split('\t')
        return [re.sub('[\n.]', '', elem) for elem in split]

    def split_into_test_train(self, data, dev_ratio, test_ratio):
        dataset = data.copy()
        num_dev = int(len(dataset) * dev_ratio)
        num_test = int(len(dataset) * test_ratio)

        dev = []
        for i in range(num_dev):
            line = dataset.pop(dataset.index(random.choice(dataset)))
            dev.append(line)

        test = []
        for i in range(num_test):
            line = dataset.pop(dataset.index(random.choice(dataset)))
            test.append(line)

        # shuffle the dataset
        shuffled_data = []
        for i in range(len(dataset)):
            choice = dataset.pop(dataset.index(random.choice(dataset)))
            shuffled_data.append(choice)

        return {'train':shuffled_data, 'dev':dev, 'test':test}

    def train(self):
        return self.dataset['train']

    def dev(self):
        return self.dataset['dev']

    def test(self):
        return self.dataset['test']


class ElemTuple:
    """
    The tuple for (ipa, orth)
    """
    def __init__(self, ipa, orth):
        self._tuple = (ipa, orth)

        self.ipa = self._tuple[0]
        self.orth = self._tuple[1]


class WordLine:
    """
    A line in a dataset: data, label
    """
    def __init__(self, input):
        seq1 = self.split_nasals(input[0])
        seq2 = self.split_nasals(input[1])

        self.data_string = '<SOS> ' + ' '.join(seq1) + ' ' + input[2] + ' <EOS>'
        self.label_string = '<SOS> ' + ' '.join(seq2) + ' <EOS>'

        self.data = self.data_string.split(' ')
        self.label = self.label_string.split(' ')

    def split_nasals(self, string):
        # split by character, but leave nasal diacritics
        split = []
        for i in range(len(string)):
            if string[i] == '̃':
                try:
                    split[i-1] += string[i]
                except IndexError:
                    pass
            else:
                split.append(string[i])
        return split


class EncoderRNN(nn.Module):
    """
    Create a context tensor from an input sequence based on indices, which are then embedded
    The embedded tensor is concatenated with the IPA features matrix
    To create an encoder for orthography, specify a features_dim of 0
    """
    def __init__(self, alphabet_size, features_dim):
        # The LSTM will be fed a (seq_length, 1, EMBEDDING_DIM + num_features) tensor
        super(EncoderRNN, self).__init__()
        self.hidden_size = EMBEDDING_DIM + features_dim
        self.embedder = nn.Embedding(alphabet_size, EMBEDDING_DIM)
        self.rnn = nn.LSTM(self.hidden_size, RNN_HIDDEN_DIM, RNN_LAYERS, bidirectional=True)

    def forward(self, input, features=None):
        # input a tensor one-hot representation of the word and the features representation
        char_embed = self.embedder(input)
        if features != None:
            concat_input = torch.cat((char_embed, features), 1)
            input_tensor = concat_input.unsqueeze(1)
        else:
            input_tensor = char_embed.unsqueeze(1)
        output, hidden = self.rnn(input_tensor)
        # Without attention:
        # first = hidden[0].view(1, 1, 100)
        # hidden = first
        return output, hidden


class Attention(nn.Module):
    """
    An implementation of Bahdanau attention
    """
    def __init__(self):
        super(Attention, self).__init__()

        self.linear1 = nn.Linear(3*RNN_HIDDEN_DIM, RNN_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(RNN_HIDDEN_DIM, 1)

    def forward(self, encoder_hss, decoder_hs):
        decoder_states = decoder_hs.expand(encoder_hss.size()[0], decoder_hs.size()[1], decoder_hs.size()[2])
        conditioned = torch.cat((encoder_hss, decoder_states), 2)
        att1 = self.relu(self.linear1(conditioned))
        attention = self.linear2(att1)
        # print(attention.size(), encoder_hss.size())
        softmax = torch.softmax(attention, dim=0)
        softmax = softmax.expand(-1, -1, 2*RNN_HIDDEN_DIM)
        softmax = softmax * encoder_hss
        return softmax.sum(0).unsqueeze(0)


class DecoderRNN(nn.Module):
    """
    Decode a context vector and generate characters
    """
    def __init__(self, alphabet, input_scale=1):
        super(DecoderRNN, self).__init__()
        self.alphabet = alphabet
        self.embedding = nn.Embedding(len(self.alphabet), EMBEDDING_DIM)
        self.rnn = nn.LSTM(EMBEDDING_DIM + 2*RNN_HIDDEN_DIM*input_scale, RNN_HIDDEN_DIM, RNN_LAYERS, bidirectional=False)
        self.hidden2char = nn.Linear(RNN_HIDDEN_DIM, len(self.alphabet))
        self.attention = Attention()

    # def forward(self, example, encoder_hs):
    #     # Takes in an example (to use the gold label), without EOS marker, and a hidden state from the encoder
    #     embedded_output = self.embedding(example[:-1])
    #     expand_hs = encoder_hs.expand(embedded_output.size()[0], 1, encoder_hs.size()[2])
    #     expand_hs = torch.squeeze(expand_hs)
    #     combined_embed = torch.cat((embedded_output, expand_hs), 1)
    #     combined_embed = torch.unsqueeze(combined_embed, 1)
    #
    #     decoder_hidden_states, output = self.rnn(combined_embed)
    #     distr = self.hidden2char(decoder_hidden_states)
    #     distr = torch.log_softmax(distr, dim=2)
    #
    #     return distr, example[1:]

    def forward(self, example, encoder_hss, sos_char=44, eos_char=43):
        # forward with attention
        embedded_output = self.embedding(example[:-1])
        embedded_output = embedded_output.unsqueeze(1)
        decoder_state = (torch.zeros(1, 1, RNN_HIDDEN_DIM), torch.zeros(1, 1, RNN_HIDDEN_DIM))
        results = []
        output_char = torch.LongTensor([sos_char])
        for i in range(embedded_output.size()[0]): # sequence_length - 1
            context = self.attention(encoder_hss, decoder_state[0])
            gold_at_i = embedded_output[i].unsqueeze(0)
            concat = torch.cat((gold_at_i, context), 2)
            output, decoder_state = self.rnn(concat, decoder_state)
            prob_dist = self.hidden2char(decoder_state[0]).log_softmax(dim=2)
            # output_value = torch.argmax(prob_dist).tolist()
            results.append(prob_dist)

        results_tensor = results[0]
        for i in range(1, len(results)):
            results_tensor = torch.cat((results_tensor, results[i]), 0)
        return results_tensor, example[1:]

    def generate(self, encoder_hss, sos_char=44, eos_char=43):
        with torch.no_grad():
            decoder_state = (torch.zeros(1, 1, RNN_HIDDEN_DIM), torch.zeros(1, 1, RNN_HIDDEN_DIM)) # supposed to be a tuple
            output_char = torch.LongTensor([sos_char]) # add <SOS>
            result = []
            output_value = 0
            for i in range(MAXWFLEN): # or: while output_symbol != '<EOS>'
                output_embedding = self.embedding(output_char)
                output_embedding = torch.unsqueeze(output_embedding, 1)
                context = self.attention(encoder_hss, decoder_state[0])
                concat = torch.cat((output_embedding, context), 2)
                output, decoder_state = self.rnn(concat, decoder_state)
                prob_dist = self.hidden2char(decoder_state[0])
                output_value = torch.argmax(prob_dist).tolist()
                output_char = torch.LongTensor([output_value])
                result.append(output_value)
                if output_value == eos_char:
                    break
                # Next steps: feed in output symbol to rnn, loop until output_symbol is <EOS>
            return result


class WordInflector(nn.Module):
    def __init__(self, embedder, features_dim):
        super(WordInflector, self).__init__()
        self.embedder = embedder
        self.encoder = EncoderRNN(self.embedder.dim, features_dim)
        self.decoder = DecoderRNN(self.embedder.alphabet)

    def get_string(self, example):
        results = [self.embedder.decode(i) for i in example]
        return ' '.join(results)

    def forward(self, example, gold, features=None):
        output, encoder_hs = self.encoder(example, features)
        return self.decoder(gold, output)

    def generate(self, data, features=None, sos_char=44, eos_char=43):
        all_results = []
        with torch.no_grad():
            for i in range(len(data)):
                if features:
                    feature_tensor = features[i]
                else:
                    feature_tensor = None
                output, encoder_hs = self.encoder(data[i], feature_tensor)
                output = self.decoder.generate(output, sos_char, eos_char)
                all_results.append(self.get_string(output))
            return all_results

    def encode(self, i):
        return self.embedder.encode(i)

    def decode(self, i):
        return self.embedder.decode(i)


class OrthWordInflector(WordInflector):
    def __init__(self):
        super(OrthWordInflector, self).__init__(OneHotEncoder(), 0)


class IPAWordInflector(WordInflector):
    def __init__(self):
        super(IPAWordInflector, self).__init__(FeatureEncoder(), 0)

    def generate(self, data, **kwargs):
        return super(IPAWordInflector, self).generate(data, None, 345, 344)


class FeaturesWordInflector(WordInflector):
    def __init__(self):
        super(FeaturesWordInflector, self).__init__(FeatureEncoder(), NUM_FEATURES)

    def generate(self, data, features):
        return super(FeaturesWordInflector, self).generate(data, features, 345, 344)

    def get_features(self, n):
        return self.embedder.get_features(n)


class CombinedWordInflector(nn.Module):
    def __init__(self):
        super(CombinedWordInflector, self).__init__()
        self.orth_embedder = OneHotEncoder()
        self.ipa_embedder = FeatureEncoder()

        self.orth_encoder = EncoderRNN(self.orth_embedder.dim, 0)
        self.features_encoder = EncoderRNN(self.ipa_embedder.dim, NUM_FEATURES)
        self.decoder = DecoderRNN(self.orth_embedder.alphabet)

    def get_string(self, example):
        results = [self.orth_embedder.decode(i) for i in example]
        return ' '.join(results)

    def forward(self, orth_example, ipa_example, gold, features):
        orth_output, orth_hs = self.orth_encoder(orth_example)
        feat_output, feat_hs = self.features_encoder(ipa_example, features)
        concat = torch.cat((orth_output, feat_output), 0)
        return self.decoder(gold, concat)

    def generate(self, orth_data, ipa_data, features):
        all_results = []
        with torch.no_grad():
            for i in range(len(orth_data)):
                feature_tensor = features[i]
                orth_output, orth_hs = self.orth_encoder(orth_data[i])
                feat_output, feat_hs = self.features_encoder(ipa_data[i], feature_tensor)
                concat = torch.cat((orth_output, feat_output), 0)
                output = self.decoder.generate(concat, 44, 43)
                all_results.append(self.get_string(output))
            return all_results

    def orth_encode(self, i):
        return self.orth_embedder.encode(i)

    def orth_decode(self, i):
        return self.orth_embedder.decode(i)

    def ipa_encode(self, i):
        return self.ipa_embedder.encode(i)

    def ipa_decode(self, i):
        return self.ipa_embedder.decode(i)

    def get_features(self, n):
        return self.ipa_embedder.get_features(n)


def create_orth_dataset(label):
    return {'data':[w.orth.data for w in dataset.dataset[label]],
            'labels':[w.orth.label for w in dataset.dataset[label]]}


def create_ipa_dataset(label):
    return {'data':[w.ipa.data for w in dataset.dataset[label]],
            'labels':[w.ipa.label for w in dataset.dataset[label]]}


if __name__=='__main__':
    dataset = Dataset(0.2, 0.2)
    wi = CombinedWordInflector()
    loss_fn = nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.Adam(wi.parameters())

    train_ipa_ds = create_ipa_dataset('train')
    dev_ipa_ds = create_ipa_dataset('dev')
    test_ipa_ds = create_ipa_dataset('test')

    train_orth_ds = create_orth_dataset('train')
    dev_orth_ds = create_orth_dataset('dev')
    test_orth_ds = create_orth_dataset('test')

    # main training loop
    for epoch in range(EPOCHS):
        tot_loss = 0

        indices = [i for i in range(len(train_ipa_ds['data']))]
        random.shuffle(indices)

        for i in range(len(train_ipa_ds['data'])):
            wi.zero_grad()

            example_orth_data = torch.LongTensor([wi.orth_encode(i) for i in train_orth_ds['data'][indices[i]]])
            example_ipa_data = torch.LongTensor([wi.ipa_encode(i) for i in train_ipa_ds['data'][indices[i]]])
            example_features = torch.LongTensor([wi.get_features(i) for i in train_ipa_ds['data'][indices[i]]])

            example_label = torch.LongTensor([wi.orth_encode(i) for i in train_orth_ds['labels'][indices[i]]])

            tag_scores, tgt = wi(example_orth_data, example_ipa_data, example_label, example_features) # (batch_size, 1, alph_size)
            tag_scores = tag_scores.permute(1, 2, 0)
            tgt = tgt.unsqueeze(0)

            loss = loss_fn(tag_scores, tgt)
            tot_loss += loss.detach().numpy()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Example %d of %d' % (i+1, len(train_ipa_ds['data'])))

        print()
        avg_loss = tot_loss / len(train_ipa_ds['data'])
        print('EPOCH %d: AVG LOSS PER EX: %.5f' % (epoch+1, avg_loss))

        dev_orth_data = [torch.LongTensor([wi.orth_encode(i) for i in datum]) for datum in dev_orth_ds['data']]
        dev_ipa_data = [torch.LongTensor([wi.ipa_encode(i) for i in datum]) for datum in dev_ipa_ds['data']]
        dev_features = [torch.LongTensor([wi.get_features(i) for i in datum]) for datum in dev_ipa_ds['data']]
        dev_labels = [' '.join(label[1:-1]) for label in dev_orth_ds['labels']]

        sys_dev_words = wi.generate(dev_orth_data, dev_ipa_data, dev_features)
        sys_dev_words = [word[:-6] for word in sys_dev_words]  # -6 for orth, -5 for ipa
        print('DEV ACC: %.2f%%' % accuracy(sys_dev_words, dev_labels))
        print()

    with open('results/combined_test_withatt.txt', 'w') as output:
        test_orth_data = [torch.LongTensor([wi.orth_encode(i) for i in datum]) for datum in test_orth_ds['data']]
        test_ipa_data = [torch.LongTensor([wi.ipa_encode(i) for i in datum]) for datum in test_ipa_ds['data']]
        test_features = [torch.LongTensor([wi.get_features(i) for i in datum]) for datum in test_ipa_ds['data']]
        test_labels = [' '.join(label[1:-1]) for label in test_orth_ds['labels']]

        sys_test_words = wi.generate(test_orth_data, test_ipa_data, test_features)
        sys_test_words = [''.join(word[:-6]) for word in sys_test_words]  # -6 for orth, -5 for ipa

        for i in range(len(sys_test_words)):
            output.write('SYS: ' + sys_test_words[i] + '\n')
            output.write('GOLD: ' + test_labels[i] + '\n')
            output.write('\n')