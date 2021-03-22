# @author Utkarsh Patel
#

from Preprocess import Preprocess
import pandas as pd
import numpy as np 

import argparse
import os
from tqdm import tqdm
import pickle

class Comment:
    '''Container storing comments and their labels as predicted by various models'''
    def __init__(self, cid, text, label):
        self.id = cid
        self.text = text
        self.label = label
        self.labels = dict()
        self.scores = dict()
        self.ah = dict()

    def __str__(self):
        s = 'id: ' + str(self.id) + '\n'
        s += 'comment: ' + self.text + '\n'
        s += 'label: ' + self.label + '\n'
        return s

    def add_model(self, model, label, score, words):
        self.labels[model] = label
        self.scores[model] = score
        self.ah[model] = words

def create_unclassified():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default=None, type=str, required=True, help="path of json file")
    parser.add_argument("--outdir", default=None, type=str, required=True, help="folder to write .log file")
    args = parser.parse_args()
    
    pr = Preprocess()

    reader = pd.read_json(args.indir, lines=True, compression=None)
    comments = list(reader['body'])
    violated_rule = list(reader['violated_rule'])

    writer_addr = os.path.join(args.outdir, 'comments_0.log')
    writer = open(writer_addr, 'wb')

    for i in tqdm(range(len(comments)), unit=' comments', desc='Comments processed: '):
        label = 'none'
        if violated_rule[i] == 2:   label = 'ah'
        cur = ' '.join(pr.preprocess(comments[i]))
        e = Comment(i, cur, label)
        print(e)
        pickle.dump(e, writer)

    writer.close()

if __name__ == '__main__':
    create_unclassified()
