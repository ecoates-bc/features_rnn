import bs4
import requests
import re

"""
Scrape a verb paradigm from wiktionary
"""

def scrape_verb(link):
    url = requests.get(link).text
    soup = bs4.BeautifulSoup(url, 'html.parser')

    french_title = soup.find('span', attrs={'id':'French'})
    inf_ipa = None
    table = None
    for elem in french_title.parent.next_siblings:
        if elem.name == 'ul':
            for bullet in elem.contents:
                for item in bullet:
                    if type(item) != str and item.name == 'span' and item.attrs['class'] == ['IPA'] and inf_ipa == None:
                        inf_ipa = re.sub('/', '', item.next)
        if elem.name == 'div' and 'class' in elem.attrs and elem.attrs['class'] == ['NavFrame']:
            for item in elem.contents:
                if item.name == 'div' and item.attrs['class'] == ['NavContent']:
                    for member in item.contents:
                        if member.name == 'table' and member.attrs['class'] == ['inflection-table'] and table == None:
                            table = member.contents[1].contents
        if elem.name == 'h2':
            break

    table = [item for item in table if item != '\n']
    infinitive_colm = table[0]
    gerund_colm = table[2]
    pp_colm = table[4]
    pres_colm = table[8]

    infinitives = []
    for elem in infinitive_colm.contents:
        if elem.name == 'td':
            infinitives.append(re.sub('\n', '', elem.next))
    infinitives.append(inf_ipa)

    gerunds = []
    for elem in gerund_colm.contents:
        if elem.name == 'td':
            for item in elem:
                if item.name == 'span' and 'class' in item.attrs and 'Latn' in item.attrs['class']:
                    gerunds.append(item.next.text)
                elif item.name == 'span':
                    if item.next.name == 'span' and item.next.attrs['class'] == ['IPA']:
                        gerunds.append(re.sub('/', '', item.next.next))

    pps = []
    for elem in pp_colm.contents:
        if elem.name == 'td':
            for item in elem:
                if item.name == 'span' and 'class' in item.attrs and 'Latn' in item.attrs['class']:
                    pps.append(item.next.text)
                elif item.name == 'span':
                    if item.next.name == 'span' and item.next.attrs['class'] == ['IPA']:
                        pps.append(re.sub('/', '', item.next.next))

    presents = {'1sg':[], '2sg':[], '3sg':[], '1pl':[], '2pl':[], '3pl':[]}
    for elem in pres_colm.contents:
        cur_pers = None
        if elem.name == 'td':
            for item in elem.contents:
                if item.name == 'span' and 'class' in item.attrs and 'Latn' in item.attrs['class']:
                    if re.search(r'^1\|s', item.attrs['class'][-1]):
                        cur_pers = '1sg'
                    elif re.search(r'^1\|p', item.attrs['class'][-1]):
                        cur_pers = '1pl'
                    elif re.search(r'2\|s', item.attrs['class'][-1]):
                        cur_pers = '2sg'
                    elif re.search(r'2\|p', item.attrs['class'][-1]):
                        cur_pers = '2pl'
                    elif re.search(r'3\|s', item.attrs['class'][-1]):
                        cur_pers = '3sg'
                    elif re.search(r'3\|p', item.attrs['class'][-1]):
                        cur_pers = '3pl'

                    presents[cur_pers].append(item.next.next)
                elif item.name == 'span':
                    if item.next.name == 'span' and item.next.attrs['class'] == ['IPA']:
                        presents[cur_pers].append(re.sub('/', '', item.next.next))

    verb_dict = {
        'orth':{
            'inf':infinitives[0],
            'ger':gerunds[0],
            'pp':pps[0],
            'pres':{
                '1sg':presents['1sg'][0],
                '2sg':presents['2sg'][0],
                '3sg':presents['3sg'][0],
                '1pl':presents['1pl'][0],
                '2pl':presents['2pl'][0],
                '3pl':presents['3pl'][0]
            }
        },
        'ipa':{
            'inf': infinitives[1],
            'ger': gerunds[1],
            'pp': pps[1],
            'pres': {
                '1sg': presents['1sg'][1],
                '2sg': presents['2sg'][1],
                '3sg': presents['3sg'][1],
                '1pl': presents['1pl'][1],
                '2pl': presents['2pl'][1],
                '3pl': presents['3pl'][1]
            }
        }
    }

    return verb_dict


def scrape_verb_page(link):
    print('New verb page #####################################################################')
    url = requests.get(link).text
    soup = bs4.BeautifulSoup(url, 'html.parser')

    colms = soup.find_all('div', attrs='mw-category-group')[-1]
    verb_list = [elem for elem in colms.contents[2] if not type(elem) is bs4.NavigableString]

    verbs = []
    for link in verb_list:
        a = link.find('a')
        link = a.get('href')
        link = 'https://en.wiktionary.org' + link

        verbs.append(scrape_verb(link))
        print(a.text)

    return verbs


def scrape_all_verbs(link, first):
    url = requests.get(link).text
    soup = bs4.BeautifulSoup(url, 'html.parser')

    links = soup.find_all('a')
    if first:
        next = links[90].get('href')
    else:
        next = links[91].get('href')
    next_link = 'https://en.wiktionary.org' + next

    all_verbs = []
    all_verbs += scrape_verb_page(link)

    try:
        next_verbs = scrape_all_verbs(next_link, False)
        all_verbs += next_verbs
    except:
        pass
    return all_verbs


verbs = [re.sub('\n', '', line) for line in open('data/verbs3.txt', 'r').readlines()]
verb_words = []
for v in verbs:
    print(v)
    try:
        verb_words.append(scrape_verb('https://en.wiktionary.org/wiki/%s#French' % v))
    except:
        pass


def make_str_form(spell, file, verb):
    base_form = verb[spell]['inf']
    inf_line = '%s\t%s\t%s\n' % (base_form, base_form, 'V;INF')
    file.write(inf_line)
    ger_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['ger'], 'V;PRES;GER')
    file.write(ger_line)
    pp_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pp'], 'V;PST;PP')
    file.write(pp_line)
    sg1_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pres']['1sg'], 'V;PRES;SG;1P')
    file.write(sg1_line)
    sg2_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pres']['2sg'], 'V;PRES;SG;2P')
    file.write(sg2_line)
    sg3_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pres']['3sg'], 'V;PRES;SG;3P')
    file.write(sg3_line)
    pl1_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pres']['1pl'], 'V;PRES;PL;1P')
    file.write(pl1_line)
    pl2_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pres']['2pl'], 'V;PRES;PL;2P')
    file.write(pl2_line)
    pl3_line = '%s\t%s\t%s\n' % (base_form, verb[spell]['pres']['3pl'], 'V;PRES;PL;3P')
    file.write(pl3_line)


with open('data/fra_verbs_ORTH.tsv', 'w') as verb_file:
    for v in verb_words:
        make_str_form('orth', verb_file, v)

with open('data/fra_verbs_IPA.tsv', 'w') as verb_file:
    for v in verb_words:
        make_str_form('ipa', verb_file, v)


# with open('data/fra_verbs_IPA.tsv', 'w') as verb_file:
#     for item in verbs:
#         if item:
#             try:
#                 verb_file.write(item['inf'][1] + '\t' + item['inf'][1] + '\t' + 'V;INF' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['ger'][1] + '\t' + 'V;GER' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pp'][1] + '\t' + 'V;PP' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pres'][0][1] + '\t' + 'V;PRES;1P;SG' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pres'][1][1] + '\t' + 'V;PRES;2P;SG' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pres'][2][1] + '\t' + 'V;PRES;3P;SG' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pres'][3][1] + '\t' + 'V;PRES;1P;PL' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pres'][4][1] + '\t' + 'V;PRES;2P;PL' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['pres'][5][1] + '\t' + 'V;PRES;3P;PL' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['fut'][1] + '\t' + 'V;FUT' + '\n')
#                 verb_file.write(item['inf'][1] + '\t' + item['cond'][1] + '\t' + 'V;COND' + '\n')
#             except:
#                 pass
#
# with open('data/fra_verbs_ORTH.tsv', 'w') as verb_file:
#     for item in verbs:
#         if item:
#             try:
#                 verb_file.write(item['inf'][0] + '\t' + item['inf'][0] + '\t' + 'V;INF' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['ger'][0] + '\t' + 'V;GER' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pp'][0] + '\t' + 'V;PP' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pres'][0][0] + '\t' + 'V;PRES;1P;SG' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pres'][1][0] + '\t' + 'V;PRES;2P;SG' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pres'][2][0] + '\t' + 'V;PRES;3P;SG' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pres'][3][0] + '\t' + 'V;PRES;1P;PL' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pres'][4][0] + '\t' + 'V;PRES;2P;PL' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['pres'][5][0] + '\t' + 'V;PRES;3P;PL' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['fut'][0] + '\t' + 'V;FUT' + '\n')
#                 verb_file.write(item['inf'][0] + '\t' + item['cond'][0] + '\t' + 'V;COND' + '\n')
#             except:
#                 pass
