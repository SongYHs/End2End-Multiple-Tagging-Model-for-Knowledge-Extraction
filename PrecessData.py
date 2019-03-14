#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:50:34 2019

@author: Song Yunhua
"""
import numpy as np
import pickle as cPickle
import json

def load_vec_pkl(fname,vocab,k=300):
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = cPickle.load(open(fname,'rb'),encoding='iso-8859-1')
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    i=0
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
            i+=1
        W[vocab[word]] = w2v[word]
    print(str(i)+' words unknow in pretrain words')
    return w2v,k,W

def make_idx_data_index_EE_LSTM(file,max_s,source_vob,target_vob):
    data_s_all=[]
    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()
    sent=json.loads(fr[0].strip('\r\n'))
    
    kk=len(sent['tags'])//len(sent['tokens'])
    max_t=kk*max_s
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        len_s=len(sent['tokens'])
        t_sent = sent['tags']
        data_t = []
        data_s = []
        if len(s_sent) > max_s:
            i=max_s-1
            while i >= 0:
                data_s.append(source_vob[s_sent[i]])
                i-=1
        else:
            num=max_s-len(s_sent)
            for inum in range(0,num):
                data_s.append(0)
            i=len(s_sent)-1
            while i >= 0:
                data_s.append(source_vob[s_sent[i]])
                i-=1
        data_s_all.append(data_s)
        if len(t_sent) > max_t:
            for i in range(kk):
                for word in t_sent[i*len_s:i*len_s+max_s]:
                    data_t.append(target_vob[word])
        else:
            for ki in range(kk):
                for word in t_sent[ki*len_s:ki*len_s+len_s]:
                    data_t.append(target_vob[word])
                for word in range(len_s,max_s):
                    data_t.append(0)
        data_t_all.append(data_t)
    f.close()
    return [data_s_all,data_t_all]

def get_word_index(train,test,labeltxt):
    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    f=open(labeltxt,'r')
    fr=f.readlines()
    end=['__E1S','__E1B','__E1I','__E1L','__E2S','__E2B','__E2I','__E2L']
    count = 2
    target_vob['O']=1
    target_idex_word[1]='O'
    for line in fr:
        line=line.strip('\n')
        if line and not line=='O':
            for lend in end:
                target_vob[line+lend]=count
                target_idex_word[count]=line+lend
                count+=1
    count = 1
    max_s=0
    f = open(train,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()
    f.close()
    f = open(test,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__()>max_s:
            max_s = sourc.__len__()
    f.close()
    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count+=1
    if not source_vob.__contains__("UNK"):
        source_vob["UNK"] = count
        sourc_idex_word[count] = "UNK"
        count+=1
    return source_vob,sourc_idex_word,target_vob,target_idex_word,max_s

def get_data_e2e(trainfile,testfile,labeltxt,w2v_file,eelstmfile,maxlen = 50):
    source_vob, sourc_idex_word, target_vob, target_idex_word, max_s = \
                get_word_index(trainfile, testfile,labeltxt)
    print("source vocab size: " + str(len(source_vob)))
    print("target vocab size: " + str(len(target_vob)))
    source_w2v ,k ,source_W= load_vec_pkl(w2v_file,source_vob)
    print("word2vec loaded!")
    print("num words in source word2vec: " + str(len(source_w2v))+\
          "source  unknown words: "+str(len(source_vob)-len(source_w2v)))
    if max_s > maxlen:
        max_s = maxlen
    print('max soure sent lenth is ' + str(max_s))
    train = make_idx_data_index_EE_LSTM(trainfile,max_s,source_vob,target_vob)
    test = make_idx_data_index_EE_LSTM(testfile, max_s, source_vob, target_vob)
    print("dataset created!")
    cPickle.dump([train,test,source_W,source_vob,sourc_idex_word,
                  target_vob,target_idex_word,max_s,k],open(eelstmfile,'wb'))
