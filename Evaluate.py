#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:59:23 2019

@author: Song Yunhua
"""

import json

def evaluavtion_triple_new(testresult,senttext,savetagfile,kk=1,max_s=50):
    total_predict_right=0.
    total_predict=0.
    total_right = 0.
    prn={}
    p={}
    r={}
    for i in range(kk):
        prn[i],p[i],r[i]=0,0,0
    f=open(savetagfile,'w')
    for i,sent in enumerate(testresult):
        tokens=senttext[i]
        ptag = sent[0]
        ttag = sent[1]
        predictrightnum, predictnum ,rightnum,prn0,p0,r0,Pretriplets,Rigtriplets=\
                            count_sentence_triple_num(ptag,ttag,tokens,kk,max_s)
        total_predict_right+=predictrightnum
        total_predict+=predictnum
        total_right += rightnum
        
        for j in range(kk):
            prn[j]+=prn0[j]
            p[j]+=p0[j]
            r[j]+=r0[j]
        newsent=dict()
        newsent['tokens']=senttext[i]
        newsent['predict tags']=ptag
        newsent['right tags']=ttag
        newsent['Predict triplets']=Pretriplets
        newsent['Right triplets']=Rigtriplets
        f.write(json.dumps(newsent)+'\n')
    f.close()
    P = total_predict_right /float(total_predict) if total_predict!=0 else 0
    R = total_predict_right /float(total_right)
    F = (2*P*R)/float(P+R) if P!=0 else 0
    return P,R,F,total_predict_right,total_predict,total_right,prn,p,r

def count_sentence_triple_num(ptag,ttag,tokens,kk=1,max_s=50):
    #transfer the predicted tag sequence to triple index
    
    predict_triplet=[]
    right_triplet=[]
    prn0={}
    p0={}
    r0={}
    for i in range(kk):
        predict_rmpair0= tag_to_triple_index_new1(ptag[i*max_s:i*max_s+max_s])
        right_rmpair0 = tag_to_triple_index_new1(ttag[i*max_s:i*max_s+max_s])
        predict_triplet0=proximity(predict_rmpair0)
        right_triplet0=proximity(right_rmpair0)
        for pt in predict_triplet0:
            if not predict_triplet.__contains__(pt):
                predict_triplet.append(pt)
        for rt in right_triplet0:
            if not right_triplet.__contains__(rt):
                right_triplet.append(rt)
        prn0[i]=0
        p0[i]=len(predict_triplet0)
        r0[i]=len(right_triplet0)
        for triplet in predict_triplet0:
            if right_triplet0.__contains__(triplet):
                prn0[i]+=1
    predict_right_num = 0       # the right number of predicted triple
    predict_num = len(predict_triplet)     # the number of predicted triples
    right_num = len(right_triplet)
    pre_relationMentions=get_triplet(predict_triplet,tokens)
    rig_relationMentions=get_triplet(right_triplet,tokens)
    for triplet in predict_triplet:
        if right_triplet.__contains__(triplet):
            predict_right_num+=1
        
    return predict_right_num,predict_num,right_num,prn0,p0,r0,pre_relationMentions,rig_relationMentions

def get_triplet(tpositions,tokens):
    rels=[]
    for triplet in tpositions:
        
        e1=triplet[0]
        label=triplet[1]
        e2=triplet[2]
        entity1=''
        for e1i in range(e1[0],e1[1]):
            entity1+=' '+tokens[e1i]
            
        entity2=''
        for e2i in range(e2[0],e2[1]):
            entity2+=' '+tokens[e2i]
        rel={"em1Text":entity1,"em2Text":entity2,"label":label}
        rels.append(rel)
    return rels

def proximity(predict_rmpair):
    triplet=[]
    for type1 in predict_rmpair:
        eelist = predict_rmpair[type1]
        e1 = eelist[0]
        e2 = eelist[1]
        if len(e2)<len(e1):
            for i in range(len(e2)):
                e2i0,e2i1=e2[i][0],e2[i][1]
                e2i=(e2i0+e2i1)/2
                mineij=100
                for j in range(len(e1)):
                    e1j=(e1[j][0]+e1[j][1])/2
                    if mineij>abs(e1j-e2i):
                        mineij=abs(e1j-e2i)
                        e1ij=e1[j]
                e1.remove(e1ij)
                triplet.append((e1ij,type1,e2[i]))
        else:
            for i in range(len(e1)):
                e1i0,e1i1=e1[i][0],e1[i][1]
                e1i=(e1i0+e1i1)/2
                mineij=100
                for j in range(len(e2)):
                    e2j=(e2[j][0]+e2[j][1])/2
                    if mineij>abs(e1i-e2j):
                        mineij=abs(e1i-e2j)
                        e2ij=e2[j]
                e2.remove(e2ij)
                triplet.append((e1[i],type1,e2ij))
    return triplet

def tag_to_triple_index_new1(ptag):
    rmpair={}
    for i in range(0,len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist=[]
                e1=[]
                e2=[]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
#                            print(j,len(ptag))
                            if ptag[j].__contains__("1") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist=rmpair[type_e[0]]
                e1=eelist[0]
                e2=eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i,i+1))
                    elif type_e[1].__contains__("B"):
                        j=i+1
                        v='B'
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and ptag[j].__contains__(type_e[0]):
                                if ptag[j].__contains__("I"):
                                    j+=1
                                    v='I'
                                elif  ptag[j].__contains__("L")  :
                                    j+=1
                                    v='L'
                                    break
                                else:
                                    break
                            else:
                                break
                        if v=='L':
                            e2.append((i, j))
                eelist[0]=e1
                eelist[1]=e2
                rmpair[type_e[0]] = eelist
    return rmpair     

