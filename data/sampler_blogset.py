import pandas as pd
import argparse
import random
from pathlib import Path
import re

parser = argparse.ArgumentParser()

parser.add_argument('--train_size', type=int, default=10,
                    help='size of sample in megabytes')
parser.add_argument('--valid_size', type=int, default=1,
                    help='size of sample in megabytes')
parser.add_argument('--test_size', type=int, default=1,
                    help='size of sample in megabytes')
parser.add_argument('--folder', type=str, default='./blogset',
                    help='folder to save sample')
parser.add_argument('--random', type=bool, default=False,
                    help='randomize dataset sample')

args = parser.parse_args()
args.tied = True


def is_not_in(string, terms):
    for term in terms:
        if term in string:
            return False
    return True


def split_punctuation(content):
    content = re.findall(r"[\w']+|[.,!?;]", str(content))
    return ' '.join(content)


def remove_selected_terms(content, max_len, terms):
    new_cont = []
    last_word = ''
    for word in content.split():
        if is_not_in(word, terms) and len(word) < max_len and (
                word != last_word):
            new_cont.append(word)
            last_word = word
    return u' '.join(new_cont)


def clean_content(content, max_len=20):
    content = content.lower()
    content = remove_selected_terms(content, 25, ['nbsp', 'http', r'\ufeff', '@', 'www'])
    content = split_punctuation(content)
    return content


header_list = ['post.id', 'blog.id', 'published', 'title', 'content', 'author.id', 'author.displayName',
               'replies.totalItems', 'tags']
blog = pd.read_csv('blogset/blogset-br.csv', encoding='utf8', usecols=['content'], chunksize=10000, names=header_list)

f = open(f'{args.folder}/train.txt', 'w', encoding='utf8')
v = open(f'{args.folder}/valid.txt', 'w', encoding='utf8')
t = open(f'{args.folder}/test.txt', 'w', encoding='utf8')
dataset = 0

for chunk in blog:
    for row in chunk.itertuples():

        train_file_size = Path(f'{args.folder}/train.txt').stat().st_size
        valid_file_size = Path(f'{args.folder}/valid.txt').stat().st_size
        test_file_size = Path(f'{args.folder}/test.txt').stat().st_size

        print('\rIndex: {} \t Train Size: {} \t Valid Size: {} \t Test Size: {}'
              .format(row.Index, train_file_size, valid_file_size, test_file_size), end='')
        row = clean_content(row.content)

        if dataset == 0:
            if (train_file_size <= args.train_size * 1000000) and (random.random() > 0.7) and (len(row) > 300):
                f.write(row + '\n')
            elif train_file_size > args.train_size * 1000000:
                f.close()
                dataset += 1
        elif dataset == 1:
            if (valid_file_size <= args.valid_size * 1000000) and (random.random() > 0.7) and (len(row) > 300):
                v.write(row + '\n')
            elif valid_file_size > args.valid_size * 1000000:
                v.close()
                dataset += 1
        elif dataset == 2:
            if (test_file_size <= args.test_size * 1000000) and (random.random() > 0.7) and (len(row) > 300):
                t.write(row + '\n')
            elif test_file_size > args.test_size * 1000000:
                t.close()
                exit()

f.close()
