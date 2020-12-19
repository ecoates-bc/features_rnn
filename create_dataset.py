import csv

"""
Create the IPA inflected/base form pairs
"""


def read_from_file(path):
    return [line[:-1].split('\t') for line in open(path, 'r').readlines()]


unimorph = read_from_file('data/legacy/fra.txt')
ipa_train = read_from_file('data/legacy/fre_train.tsv')
ipa_dev = read_from_file('data/legacy/fre_dev.tsv')
ipa_test = read_from_file('data/legacy/fre_test.tsv')


def find_word_in_unimorph(word):
    for entry in unimorph:
        try:
            if entry[1] == word:
                return {'word':entry[1], 'bf':entry[0], 'form':entry[2]}
        except IndexError:
            pass


def get_ipa_transcr(word):
    for entry in ipa_train:
        if entry[0] == word:
            return entry[1]
    for entry in ipa_dev:
        if entry[0] == word:
            return entry[1]
    for entry in ipa_test:
        if entry[0] == word:
            return entry[1]


def create_ds(category):
    dataset = []
    for line in category:
        print(line[0])
        try:
            word = find_word_in_unimorph(line[0])
            word['word'] = get_ipa_transcr(word['word'])
            word['bf'] = get_ipa_transcr(word['bf'])

            if word['bf'] is not None:
                dataset.append(word)
        except TypeError:
            pass

    return dataset


test_ds = create_ds(ipa_train)
with open('data/legacy/train_ds.txt', 'w') as outf:
    for word in test_ds:
        outf.write(word['word'] + '\t' + word['bf'] + '\t' + word['form'] + '\n')
