#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:29:08 2019

@author: Song Yunhua
"""
import json,unicodedata,nltk


def tag_sent(source_json,tag_json,max_token=50):
    train_json_file = open(tag_json, 'w')
    file = open(source_json, 'r')
    sentences_0 = file.readlines()
    c=0
    Tkk=[]
    ii=0
    print(source_json)
    print(len(sentences_0))
    Tkk={}
    vV=[]
    Mlabel={}
    count_r={}
    for line in sentences_0:
        c+=1
        kk=1
        count_r[c-1]=0
        Tkk[c-1]=0
        if not c%10000:
            print(c)
        sent = json.loads(line.strip('\r\n'))
        flag=0
        sentText = str(unicodedata.normalize('NFKD', sent['sentText']).encode('ascii','ignore'),'ascii').rstrip('\n').rstrip('\r')
        tags=[]
#        tokens = nltk.word_tokenize(sentText)        #0311
        tokens0 = nltk.word_tokenize(sentText)        #0311
        tokens=tokens0[:min(max_token,len(tokens0))]  #0311
        for i in range(0,len(tokens)):
            tags.append('O')
        emIndexByText = {}
        for em in sent['entityMentions']:
            emText = unicodedata.normalize('NFKD', em['text']).encode('ascii','ignore')
            tokens1=tokens
            em1=emText.split()
            flagE=True
            if emIndexByText.__contains__(emText):
                flagE=False
            while flagE:
                start, end = find_index(tokens1, em1)
                if start!=-1 and end!=-1:
                    tokens1=tokens1[end:]
                    if emText not in emIndexByText:
                        emIndexByText[emText]=[(start,end)]
                    elif not emIndexByText[emText].__contains__((start, end)):
                        offset = emIndexByText[emText][-1][1]
                        emIndexByText[emText].append((start+offset, end+offset))
                else:
                    break                    
        for rm in sent['relationMentions']:
            if not rm['label'].__eq__('None'):
                rmlabel=rm["label"]
                em1 = unicodedata.normalize('NFKD', rm['em1Text']).encode('ascii','ignore')
                em2 = unicodedata.normalize('NFKD', rm['em2Text']).encode('ascii','ignore')
                if emIndexByText.__contains__(em1) and emIndexByText.__contains__(em2):
                    ind1=emIndexByText[em1]
                    ind2=emIndexByText[em2]
                    minind=len(tokens)
                    labelindex=[]
                    for i1ind,i1 in enumerate(ind1):
                        for i2ind,i2 in enumerate(ind2):
                            if (i2[0]-i1[1])*(i2[1]-i1[0])>0:
                                if minind>abs(i2[1]-i1[1]):
                                    minind=abs(i2[1]-i1[1])
                                    labelindex=[i1ind,i2ind]
                    if labelindex:
                        i1ind=labelindex[0]
                        i2ind=labelindex[1]
                        start1=ind1[i1ind][0]
                        end1  =ind1[i1ind][1]
                        start2=ind2[i2ind][0]
                        end2  =ind2[i2ind][1]
                        tag1Previous=[]
                        tag2Previous=[]
                        if end1 - start1 == 1:
                            tag1Previous.append(rmlabel+"__E1S")
                        elif end1 - start1 == 2:
                            tag1Previous.append( rmlabel + "__E1B")
                            tag1Previous.append(rmlabel + "__E1L")
                        else:
                            tag1Previous.append(rmlabel + "__E1B")
                            for ei in range(start1+1,end1-1):
                                tag1Previous.append(rmlabel + "__E1I")
                            tag1Previous.append(rmlabel + "__E1L")
                        if end2 - start2 == 1:
                            tag2Previous.append(rmlabel+"__E2S")
                        elif end2 - start2 == 2:
                            tag2Previous.append( rmlabel + "__E2B")
                            tag2Previous.append(rmlabel + "__E2L")
                        else:
                            tag2Previous.append(rmlabel + "__E2B")
                            for ei in range(start2+1,end2-1):
                                tag2Previous.append(rmlabel + "__E2I")     
                            tag2Previous.append(rmlabel + "__E2L")
                        while True:    
                            valid1=True
                            vT1=0
                            for ei in range(start1,end1):
                                if not tags[ei].__eq__('O'):
                                    valid1 = False
                                    break
                            if not valid1:
                                valid1=True
                                vT1=1
                                for ei in range(start1,end1):
                                    if not tags[ei].__eq__(tag1Previous[ei-start1]):
                                        valid1 = False
                                        vT1=0
                                        break
                            valid2=True
                            vT2=0
                            for ei in range(start2,end2):
                                if not tags[ei].__eq__('O'):
                                    valid2 = False
                                    break
                            if not valid2:
                                valid2=True
                                vT2=1
                                for ei in range(start2,end2):
                                    if not tags[ei].__eq__(tag2Previous[ei-start2]):
                                        valid2 = False
                                        vT2=0
                                        break
                            if  valid1 and valid2:
                                for ei in range(start2,end2):
                                    tags[ei]=tag2Previous[ei-start2]
                                for ei in range(start1,end1):
                                    tags[ei]=tag1Previous[ei-start1]
                                Tkk[c-1]=kk
                                if not Mlabel.__contains__(rmlabel):
                                    Mlabel[rmlabel]=[c-1]
                                else:
                                    Mlabel[rmlabel].append(c-1)
                                if not (vT1 and vT2):
                                    ii+=1
                                    count_r[c-1]+=1
                                flag=1
                                if (vT1 or vT2) and not (vT1 and vT2):
                                    vV.append(c-1)
                                break
                            else:
                                start1+=len(tokens)
                                end1+=len(tokens)
                                start2+=len(tokens)
                                end2+=len(tokens)
                            if end2>kk*len(tokens):
                                kk+=1
                                for ki in range(len(tokens)):
                                    tags.append('O')
        if 1:
            newsent = dict()
            newsent['tokens'] = tokens
            newsent['tags'] = tags
            newsent['lentags/lentokens']=kk*flag
            train_json_file.write(json.dumps(newsent)+'\n')
    train_json_file.close()
    return Tkk,vV,ii,Mlabel,count_r
def datakk(file0,file1,kk=1,isTrain=True):   
    fread=open(file0,'r')
    sentence=fread.readlines()
    fwrite=open(file1,'w')
    ii=0
    print('Origin Sentence:'+str(len(sentence)))
    for line in sentence:
        sent=json.loads(line)
        tkk=sent['lentags/lentokens']
        tkk=len(sent['tags'])//len(sent['tokens'])
        lent=len(sent['tokens'])
        if kk==1:
            for i in range(tkk):
                newsent=dict()
                newsent['tokens']=sent['tokens']
                newsent['tags']=sent['tags'][i*lent:i*lent+lent]
                ii+=1
                fwrite.write(json.dumps(newsent)+'\n')
                
        elif kk==2:
            if tkk>=2:
                for i in range(tkk):
                    for j in range(i+1,tkk):
                        newsent=dict()
                        newsent['tokens']=sent['tokens']
                        newsent['tags']=sent['tags'][i*lent:i*lent+lent]
                        newsent['tags'].extend(sent['tags'][j*lent:j*lent+lent])
                        fwrite.write(json.dumps(newsent)+'\n')
                        ii+=1
            else:
                newsent=dict()
                newsent['tokens']=sent['tokens']
                newsent['tags']=sent['tags']
                newsent['tags'].extend(['O' for i in sent['tokens']])
                fwrite.write(json.dumps(newsent)+'\n')
                ii+=1
        elif kk==3:
            for _ in range(kk-tkk):
                sent['tags'].extend(['O' for i in sent['tokens']]) 
                tkk=3
            for i in range(tkk):
                for j in range(i+1,tkk):
                    for k in range(j+1,tkk):
                        newsent=dict()
                        newsent['tokens']=sent['tokens']
                        newsent['tags']=sent['tags'][i*lent:i*lent+lent]
                        newsent['tags'].extend(sent['tags'][j*lent:j*lent+lent])
                        newsent['tags'].extend(sent['tags'][k*lent:k*lent+lent])
                        fwrite.write(json.dumps(newsent)+'\n')
                        ii+=1
        elif kk==4:
            for _ in range(kk-tkk):
                sent['tags'].extend(['O' for i in sent['tokens']])  
                tkk=4
            for i in range(tkk):
                for j in range(i+1,tkk):
                    for k in range(j+1,tkk):
                        for m in range(k+1,tkk):
                            newsent=dict()
                            newsent['tokens']=sent['tokens']
                            newsent['tags']=sent['tags'][i*lent:i*lent+lent]
                            newsent['tags'].extend(sent['tags'][j*lent:j*lent+lent])
                            newsent['tags'].extend(sent['tags'][k*lent:k*lent+lent])
                            newsent['tags'].extend(sent['tags'][m*lent:m*lent+lent])
                            fwrite.write(json.dumps(newsent)+'\n')
                            ii+=1
        elif kk==8:
            newsent=dict()
            newsent['tokens']=sent['tokens']
            newsent['tags']=sent['tags']
            for j in range(kk-tkk):
                newsent['tags'].extend(['O' for i in sent['tokens']])
            fwrite.write(json.dumps(newsent)+'\n')
            ii+=1
    fread.close()
    fwrite.close()
    print("lenght of "+file1+' = '+str(ii))

def get_label(infile1,infile2,labeltxt):
    file1 = open(infile1, 'r')
    file2 = open(infile2, 'r')
    labelset=[]
    sentences_0 = file1.readlines()
    sentences_1 = file2.readlines()
    for line in sentences_0+sentences_1:
        sent = json.loads(line.strip('\r\n'))
        rel=sent['relationMentions']
        for re in rel:
            label=re['label']
            if not labelset.__contains__(label) and not label.__eq__('None'):
                labelset.append(label)
    file1.close()
    file2.close()
    f=open(labeltxt,'w')
    for label in labelset:
        f.write(label+'\n')
    f.close()

def no_overlap(index11,index12,index21,index22):
    if index11>=index22:
        return True
    if index21>=index12:
        return True
    return False

def find_index(sen_split, word_split):
    index1 = -1
    index2 = -1
    for i in range(len(sen_split)):
        if str(sen_split[i]) == str(word_split[0],'ascii'):
            flag = True
            k = i
            for j in range(len(word_split)):
                if str(word_split[j],'ascii')!= sen_split[k]:
                    flag = False
                if k < len(sen_split) - 1:
                    k+=1
            if flag:
                index1 = i
                index2 = i + len(word_split)
                break
    return index1, index2