#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:43:34 2019

@author: Song Yunhua
"""
from TaggingScheme import tag_sent,datakk,get_label
from PrecessData import get_data_e2e

from Encoder_Decoder import LSTM_Decoder,GRU_Decoder,ReverseLayer2
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU,LSTM
from keras.layers.core import  TimeDistributedDense, Dropout, Activation,Merge

from Evaluate import evaluavtion_triple_new


import keras.backend as K
import pickle as cPickle 
import numpy as np
import argparse,json,time,os


def get_training_batch_xy_bias(inputsX, inputsY, max_s, max_t,
                          batchsize, vocabsize, target_idex_word,lossnum,shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        x = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        for idx, s in enumerate(excerpt):
            x[idx,] = inputsX[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                wordstr=''
                if word!=0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2,] = targetvec
        yield x, y        

def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)

def Loss(y_true,y_pred):
    loss0=K.zeros_like(y_true[:,:1,0])
    for i in range(kks):
        p1=y_pred[:,i*max_tokens:i*max_tokens+max_tokens,:]
        w1=K.sum(y_true[:,i*max_tokens:i*max_tokens+max_tokens,2:],axis=-1)
        lossi=K.zeros_like(y_true[:,:max_tokens,0])
        for j in range(kks):
            p2=y_pred[:,j*max_tokens:j*max_tokens+max_tokens,:]
            l=K.mean(K.square(p1-p2),axis=-1)*w1.any(axis=-1,keepdims=True)
            w2=K.sum(y_true[:,j*max_tokens:j*max_tokens+max_tokens,2:],axis=-1)
            lossi+=l*w2.any(axis=-1,keepdims=True)
        loss0=K.concatenate((loss0,lossi),axis=-1)
    return K.categorical_crossentropy(y_pred,y_true)-beta*loss0[:,1:]#-loss[:,1:]#


def creat_binary_tag_LSTM( sourcevocabsize,targetvocabsize, source_W,input_seq_lenth ,output_seq_lenth ,
    hidden_dim ,emd_dim,cell='lstm1',loss=Loss,optimizer='rmsprop'):#'categorical_crossentropy'
    encoder_a = Sequential()
    encoder_b = Sequential()
    l_A_embedding = Embedding(input_dim=sourcevocabsize+1,
                        output_dim=emd_dim,
                        input_length=input_seq_lenth,
                        mask_zero=True,
                        weights=[source_W])
    encoder_a.add(l_A_embedding)
    encoder_a.add(Dropout(0.3))
    encoder_b.add(l_A_embedding)
    encoder_b.add(Dropout(0.3))
    Model = Sequential()
    if cell.__contains__('lstm'):
        encoder_a.add(LSTM(hidden_dim,return_sequences=True))
        encoder_b.add(LSTM(hidden_dim,return_sequences=True,go_backwards=True))
        decodelayer=LSTM_Decoder
    else:
        encoder_a.add(GRU(hidden_dim,return_sequences=True))
        encoder_b.add(GRU(hidden_dim,return_sequences=True,go_backwards=True))
        decodelayer=GRU_Decoder
    encoder_rb = Sequential()
    encoder_rb.add(ReverseLayer2(encoder_b))#在decodelayer.py的Reverselayer2注意可学习变量
    encoder_ab=Merge(( encoder_a,encoder_rb),mode='concat')
    kk=int(cell[-1])
    if kk>1:
        a=[]
        for i in range(kk):
            decoder_i=Sequential()
            decoder_i.add(encoder_ab)
            decoder_i.add(decodelayer(hidden_dim=hidden_dim, output_dim=hidden_dim
                                             , input_length=input_seq_lenth,
                                             output_length=output_seq_lenth,
                                             state_input=False,
                                             return_sequences=True))
            a.append(decoder_i)
        decoder=Merge(tuple(a),mode='concat',concat_axis=-2)
        Model.add(decoder)
    else:
        Model.add(encoder_ab)
        Model.add(decodelayer(hidden_dim=hidden_dim, output_dim=hidden_dim
                                             , input_length=input_seq_lenth,
                                             output_length=output_seq_lenth,
                                             state_input=False,
                                             return_sequences=True)) 
    Model.add(TimeDistributedDense(targetvocabsize+1))
    Model.add(Activation('softmax'))
    Model.compile(loss=loss, optimizer=optimizer)
    return Model


def test_model(nn_model,testdata,index2word,sent_i2w,resultfile='',rfile=''):
    index2word[0]=''
    testx = np.asarray(testdata[0],dtype="int32")
    testy = np.asarray(testdata[1],dtype="int32")
    sent_i2w[0]='UNK'
    senttext=[]
    for sent in testx:
        stag=[]
        for wi in range(len(sent)):
            token=sent_i2w[sent[-wi-1]]
            stag.append(token)
        senttext.append(stag)
    batch_size=len(testdata[0])
    testlen = len(testx)
    testlinecount=0
    if len(testx)%batch_size ==0:
        testnum = len(testx)/batch_size
    else:
        extra_test_num = batch_size - len(testx)%batch_size
        extra_data = testx[:extra_test_num]
        testx=np.append(testx,extra_data,axis=0)
        extra_data = testy[:extra_test_num]
        testy=np.append(testy,extra_data,axis=0)
        testnum = len(testx)/batch_size
    testresult=[]
    for n in range(0,int(testnum)):
        xbatch = testx[n*batch_size:(n+1)*batch_size]
        ybatch = testy[n*batch_size:(n+1)*batch_size]
        predictions = nn_model.predict(xbatch)

        for si in range(0,len(predictions)):
            if testlinecount < testlen:
                sent = predictions[si]
                ptag = []
                for word in sent:
                    next_index = np.argmax(word)
                    #if next_index != 0:
                    next_token = index2word[next_index]
                    ptag.append(next_token)
                senty = ybatch[si]
                ttag=[]
                for word in senty:
                    next_token = index2word[word]
                    ttag.append(next_token)
                result = []
                result.append(ptag)
                result.append(ttag)
                testlinecount += 1
                testresult.append(result)
    cPickle.dump(testresult,open(resultfile,'wb'))
    max_s=50
    kk=len(testy[0])//len(testx[0])
    P2, R2, F2,numpr2,nump2,numr2,numprSeq,numpSeq,numrSeq = evaluavtion_triple_new(\
                                        testresult,senttext,rfile,kk,max_s)
    print ('new2 way',P2, R2, F2,numpr2,nump2,numr2)
    if kk>1:
        print('Seg result:')
        for i in range(kk):
            print(numprSeq[i],numpSeq[i],numrSeq[i])
    return P2,R2,F2

def train_e2e_model(eelstmfile, modelfile,resultdir,cell,npochos,
                    lossnum=1,batch_size = 256,retrain=False):
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, target_idex_word, max_s, k \
        = cPickle.load(open(eelstmfile, 'rb'),encoding='iso-8859-1')
    nn_model = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                    source_W=source_W, input_seq_lenth=max_s, output_seq_lenth=max_s,
                                    hidden_dim=k, emd_dim=k,cell=cell)
    if retrain:
        nn_model.load_weights(modelfile)
    P, R, F= test_model(nn_model, testdata, target_idex_word,sourc_idex_word,resultdir+'result-0',rfile=resultdir+'triplet0.json')
    x_train = np.asarray(traindata[0], dtype="int32")
    y_train = np.asarray(traindata[1], dtype="int32")
    epoch = 1
    save_inter = 1
    saveepoch = save_inter
    maxF=F
    kk=int(cell[-1])
    max_t=kk*max_s
    len_target_vob=len(target_vob)
    bat=batch_size#len(x_train)//30-1
    nn_model.optimizer.lr=0.001
    prf=[]
    print('start train')
    while (epoch < npochos):
        epoch = epoch + 1
        t=time.time()
        if not epoch%20:
            nn_model.optimizer.lr=nn_model.optimizer.lr/10.0
        for x, y in get_training_batch_xy_bias(x_train[:], y_train[:], max_s, max_t,
                                          bat, len_target_vob,
                                            target_idex_word,lossnum,shuffle=True):                
            nn_model.fit(x, y, batch_size=batch_size,
                         nb_epoch=1, show_accuracy=True, verbose=0)
        if epoch > saveepoch:
            saveepoch += save_inter
            resultfile = resultdir+"result-all"+str(saveepoch)
            print('Result of All testdata')
            P, R, F= test_model(nn_model, testdata, target_idex_word,sourc_idex_word,resultfile,rfile=resultdir+"triplet"+str(saveepoch)+'.json')
            prf.append([P,R,F])
            if F>=maxF:
                save_model(nn_model, modelfile.split('.pkl')[0]+str(saveepoch)+'.pkl')
        print('epoch '+str(epoch-1)+' costs time:'+str(time.time()-t)+'s')
    return nn_model,prf
def infer_e2e_model(eelstmfile, modelfile,resultdir,cell,predictfile):
    traindata, testdata, source_W, source_vob, sourc_idex_word, target_vob, \
        target_idex_word, max_s, k = cPickle.load(open(eelstmfile, 'rb'))
    
    nnmodel = creat_binary_tag_LSTM(sourcevocabsize=len(source_vob),targetvocabsize= len(target_vob),
                                    source_W=source_W,input_seq_lenth= max_s,output_seq_lenth= max_s,
                                    hidden_dim=k, emd_dim=k,cell='lstm1')
    nnmodel.load_weights(modelfile)
    f = open(predictfile,'r')
    fr = f.readlines()
    data_s_all=[]
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
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
    f.close()
    predictions = nnmodel.predict(data_s_all)
    testresult=[]
    for si in range(0,len(predictions)):
        sent = predictions[si]
        ptag = []
        for word in sent:
            next_index = np.argmax(word)
            #if next_index != 0:
            next_token = target_idex_word[next_index]
            ptag.append(next_token)
        testresult.append(ptag)
    cPickle.dump(testresult,open(resultdir+cell+'/result','wb'))
parser = argparse.ArgumentParser()
parser.add_argument('-train', dest='trainfile', type=str, default="./data/train.json")
parser.add_argument('-test', dest='testfile', type=str, default="./data/test.json")
parser.add_argument('-l', dest='label', type=str, default="./data/label.txt")
parser.add_argument('-w', dest='w2v', type=str, default="./data/w2v.pkl")
parser.add_argument('-e', dest='e2edatafile', type=str, default="./data/e2edata_0311_k3.pkl")
parser.add_argument('-m', dest='modeldir', type=str, default="./data/new/model_0_lstm_2/")
parser.add_argument('-r', dest='resultdir', type=str, default="./data/new/result_0_lstm_2/")
parser.add_argument('-p', dest='predictfile', type=str, default="./data/demo/result_1_lstm/predict.json")
parser.add_argument('-mode', dest='mode', type=str, default='Train', choices=['Train', 'Predict'],help='1 for train, 0 for predict')
parser.add_argument('-cell', dest='cell', type=str, default='lstm3', help='cell+k: lstm or gru+ 1,2,3')
parser.add_argument('-alpha', type=str, default="5 10 20 50", help='for test')
parser.add_argument('-beta', type=str, default="10 1 0.1 0.01", help='for test')
parser.add_argument('-epoch', type=int, default=30, help='for test')
args = parser.parse_args()
kks=int(args.cell[-1])
max_tokens=50


if __name__=="__main__":
    mode=args.mode
    alphas = args.alpha
    betas = args.beta
    labelfile=args.label
    originTrainfile = args.trainfile
    originTestfile = args.testfile
    e2edatafile = args.e2edatafile
    w2vfile=args.w2v
    global beta
    if not os.path.exists(args.modeldir):
        os.makedirs(args.modeldir)
    if not os.path.exists(args.resultdir):
        os.makedirs(args.resultdir)
    if not os.path.exists(e2edatafile):
        print ("Precess lstm data....")
        minkkTagTrainfile = originTrainfile[:-5]+'_all.json'
        minkkTagTestfile  = originTestfile[:-5]+'_all.json'
        TagTrainfile      = originTrainfile[:-5]+'Tag_'+str(kks)+'.json'
        TagTestfile       = originTestfile[:-5]+'Tag_'+str(kks)+'.json'
        _= tag_sent(originTrainfile,minkkTagTrainfile)
        _= tag_sent(originTestfile, minkkTagTestfile)
        datakk(minkkTagTrainfile,TagTrainfile,kks,isTrain=True)
        datakk(minkkTagTestfile ,TagTestfile ,kks,isTrain=False)
        get_label(originTrainfile,originTestfile,labelfile)
        get_data_e2e(TagTrainfile,TagTestfile,labelfile,w2vfile,e2edatafile,maxlen=max_tokens)
    if args.mode=='Train':
        print("Training EE model....cell="+args.cell+'\n'+'datafile:'+e2edatafile)
        PRFs=[]
        for a in alphas.split():
            alpha=int(a)
            for b in betas.split():
                beta=float(b)
                print('alpha,beta = ',alpha,beta)
                modeldir=args.modeldir+'alpha'+a+'beta'+b+'/'
                resultdir=args.resultdir+'alpha'+a+'beta'+b+'/'
                if not os.path.exists(modeldir):
                    os.makedirs(modeldir)
                if not os.path.exists(resultdir):
                    os.makedirs(resultdir)
                nn_model,PRF=train_e2e_model(e2edatafile, modeldir+'e2edata_'+args.cell+'.pkl',resultdir,args.cell,
                     npochos=args.epoch,lossnum=alpha,retrain=False)
                PRFs.append(PRF)
                cPickle.dump(PRF,open(args.resultdir+'PRF_'+args.cell+'_alpha'+\
                     str(alpha)+'_beta'+str(beta)+'.pkl','wb'))
        cPickle.dump([alphas,betas,PRFs],open(args.resultdir+'AllPRF_'+args.cell+'_alpha'+\
                     alphas+'_beta'+betas+'.pkl','wb'))
    else:
        infer_e2e_model(e2edatafile, args.modeldir+'e2edata'+args.cell+'.pkl',args.resultdir,args.cell,args.predictfile)
        
