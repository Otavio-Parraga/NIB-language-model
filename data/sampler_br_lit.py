import os
import re
DATA_PATH = './br_lit/books'

def remove_web_tokens(sentence):
    to_return = []
    for token in sentence.split():
        if ('http' not in token) and ('www' not in token):
            to_return.append(token)
    return ' '.join(to_return)

def split_punctuations(sentence):
    sentence = re.findall(r"[\w'-]+|[.,!?;]", sentence)
    return ' '.join(sentence)

def remove_non_utf8(sentence):
    return sentence.encode('utf-8', errors='ignore').decode()

def clean_line(line):
    line = remove_web_tokens(line)
    line = remove_non_utf8(line)
    line = split_punctuations(line)
    return line

books = os.listdir(DATA_PATH)

train = open(f'br_lit/train.txt', 'w+', encoding='utf8')
valid = open(f'br_lit/valid.txt', 'w+', encoding='utf8')
test = open(f'br_lit/test.txt', 'w+', encoding='utf8')

for book in books:
    with open(f'{DATA_PATH}/{book}', 'r', encoding='utf-8') as f_in:
        for line in f_in.readlines():
            for i,line in enumerate(lines):
                if i < (len(lines) * 0.8):
                    train.write(line)
                elif i < (len(lines) * 0.9):
                    valid.write(line)
                else:
                    test.write(line)
    f_in.close()    

train.close()
valid.close()
test.close()