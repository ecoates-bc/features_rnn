import epitran

"""
Use Epitran to create an additional IPA dataset, to compare to the IPA data directly from Wiktionary
"""

orth_dataset = [line for line in open('data/fra_verbs_ORTH.tsv').readlines()]


def transliterate_lines(dset):
    return_set = []
    epi = epitran.Epitran('fra-Latn')
    for line in dset:
        split = line.split('\t')
        new_line = '\t'.join([epi.transliterate(split[0], ligatures=True), epi.transliterate(split[1], ligatures=True),
                              split[2]])
        return_set.append(new_line)

    return return_set


trans = transliterate_lines(orth_dataset)

with open('data/fra_verbs_EPITRAN.tsv', 'w') as out_file:
    for line in trans:
        out_file.write(line)