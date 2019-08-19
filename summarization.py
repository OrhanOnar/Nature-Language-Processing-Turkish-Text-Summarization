#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import rouge
from rouge.rouge import rouge_n_sentence_level
from rouge.rouge import rouge_l_sentence_level
from rouge.rouge import rouge_w_sentence_level
from sklearn.svm import SVC
from sklearn import datasets, linear_model
from sklearn.feature_extraction.text import CountVectorizer
import re, math
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import newsListParts
import sys
import random

# In[4]:


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    WORD = re.compile(r'\w+')
    words = WORD.findall(text)
    return Counter(words)

def js(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    #print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


# In[5]:

f = open("news.txt", "r", encoding='utf8')
allNews = f.read().split("~")
haberler = []
for news in allNews:
    parts = news.split("*")    #haberi ögelerine ayır
    cumle = []
    
    for part in parts:
        #print("part: "+ part)
        #cumle += [part.split(".")]
        sentences = part.split(".")
        cumle += [sentences]
        
    haberler+=[cumle]
#print(haberler)
newsParts = []
suw = ['söyledi', 'açıkladı', 'belirtti', 'dedi', 'açıkla', 'açıklama', 'ifa', 'vurguladı', 'bildir', 'ekledi']
newsParts = newsListParts.newsList
labelBySentences = []
featuresBySentences = []
features = []
label= []


# In[6]:


def prepare(anlatim, tf, konum, baslik, varlik):
    global newsParts
    global labelBySentences
    global featuresBySentences
    global features
    global label 
    newsParts = newsListParts.newsList
    labelBySentences = []
    featuresBySentences = []
    features = []
    label=[]
    newsC = len(newsParts)
    for x in range(0,newsC):
        skorlar=[]
        for rSentence in newsParts[x][2]:
            rSentence = ' '.join(rSentence)
            vector1 = text_to_vector(rSentence)
            #print(rSentence)
            skor = []
            for tSentence in newsParts[x][4]:
                vector2 = text_to_vector(' '.join(tSentence))
                cosine = get_cosine(vector1, vector2)
                #print ('Cosine:', cosine)
                skor += [cosine]
            skorlar += [skor]
        maxList = []
        maxIdxList = []
        n = len(skorlar)
        for i in range(0,n):
            maxList += [max(skorlar[i])]
            maxIdxList += [skorlar[i].index(max(skorlar[i]))]
            for j in range(i+1,n):
                if(skorlar[j].index(max(skorlar[j]))==maxIdxList[i]):
                    skorlar[j][skorlar[j].index(max(skorlar[j]))]=0
        #print(skorlar)
        #print(maxIdxList)
        localLabels = []
        for i in range(0,len(newsParts[x][4])):
            if(i in maxIdxList):
                label+=[1]
                localLabels+=[1]
            else:
                label+=[0]
                localLabels+=[0]
        labelBySentences+=[localLabels]
        tfS = []
        for rSentence in newsParts[x][4]:
            rSentence = ' '.join(rSentence)
            tfS += [rSentence]
        vectorizer = CountVectorizer()
        transformed_data = vectorizer.fit_transform(tfS)

        i = len(vectorizer.get_feature_names())
        dic = vectorizer.get_feature_names()
        tfl = (transformed_data.sum(axis=0)).tolist()[0]
        fTf = []
        for rSentence in newsParts[x][4]:
            sm = 0
            wc = len(rSentence)
            for word in rSentence:
                if(word in dic):
                    sm += tfl[dic.index(re.sub(r'[^\w\s]','',word))]
            fTf += [sm/wc]
        #print(fTf)

        localFeatures = []
        for rSentence in newsParts[x][4]:
            feature = []
            if(tf==1):
                feature = [fTf[newsParts[x][4].index(rSentence)]]
            #feature = []
            #feature = []or newsParts[x][2].index(rSentence)== len(newsParts[x][2])-1or newsParts[x][2].index(rSentence)==1
            '''if(newsParts[x][4].index(rSentence)==0  ):
                feature += [1]
            else:
                feature += [0]'''
            if(konum==1):
                feature += [newsParts[x][4].index(rSentence)]
                
            if(anlatim==1):
                euw=0
                for uw in suw:
                    if(uw in rSentence):
                        euw=1
                feature += [euw]
                euw=0
                if('"' in word):
                    euw=1
                feature += [euw]
            if(varlik==1):
                feature += [newsParts[x][5][newsParts[x][4].index(rSentence)]]
            if(baslik==1):
                feature += [js(rSentence,newsParts[x][0][0])]
            features += [feature]
            localFeatures += [feature]
        featuresBySentences += [localFeatures]


# In[7]:


def compare(sysSummaryList, refSummaryList, indeks, yaz):
    sysSummary = []
    realSS = []
    refSummary = []
    realRS = []
    #print(str(len(haberler[yaz][3])) + "***"  + "***" + str(len(newsParts[indeks][2] ) ))
    for i in sysSummaryList:
        if(i==1):
            sysSummary += [1]

    for i in refSummaryList:
        if(i==1):
            refSummary += [1]
    indicesS = [i for i, x in enumerate(sysSummaryList) if x == 1]
    indicesR = [i for i, x in enumerate(refSummaryList) if x == 1]
    #print(indicesS)
    for idx in indicesS:
        sysSummary += newsParts[indeks][4][idx]
        if(indeks==yaz):
            realSS += haberler[yaz][2][idx]
    for idx in indicesR:
        refSummary += newsParts[indeks][4][idx]
        '''if(indeks==yaz):
            realRS += haberler[yaz][1][idx]'''
    refSummary = []
    for a in newsParts[indeks][2]:
        refSummary+=a
    if(indeks==yaz):
        #print(sysSummary)
        #print(refSummary)
        print("".join(realSS))
        print("".join(realRS))
    recall, precision, rouge = rouge_n_sentence_level(sysSummary, refSummary,1)

    #print('ROUGE-2-R', recall)
    #print('ROUGE-2-P', precision)
    #print('ROUGE-2-F', rouge)
    return [recall, precision, rouge]


# In[8]:


#baseline metot
def baselineCompExp(indeks,c0,c1,c2,c3,c4,yaz):
    skorlar=[]
    newsC = len(newsParts)
    cumleSayisi = len(newsParts[indeks][4])
    ozetCS = int(round(cumleSayisi/8.15183))
    if(ozetCS==0):
        ozetCS=1
    for x in range(indeks,indeks+1):
        tfS = []
        for rSentence in newsParts[x][4]:
            rSentence = ' '.join(rSentence)
            tfS += [rSentence]
        vectorizer = CountVectorizer()
        transformed_data = vectorizer.fit_transform(tfS)

        i = len(vectorizer.get_feature_names())
        dic = vectorizer.get_feature_names()
        tfl = (transformed_data.sum(axis=0)).tolist()[0]
        fTf = []
        for rSentence in newsParts[x][4]:
            sm = 0
            wc = len(rSentence)
            for word in rSentence:
                if(word in dic):
                    sm += tfl[dic.index(re.sub(r'[^\w\s]','',word))]
            fTf += [sm/wc]

        
        for rSentence in newsParts[x][4]:
            skor = []
            jac = js(rSentence,newsParts[x][0][0])
            #print (tSentence ,' :js: ', jac)
            skor += [jac]
            skor += [fTf[newsParts[x][4].index(rSentence)]]
            if(newsParts[x][4].index(rSentence)==0 ):
                skor += [1]
            else:
                skor += [0]
            skor += [js(vectorizer.get_feature_names(),rSentence)] #centerality
            skor += [newsParts[x][5][newsParts[x][4].index(rSentence)]]
            skorlar += [skor]
    #uzunluk = len(skorlar)
    top=[]
    for skor in skorlar: #0 title; 1 tf; 2 konum; 3 cente; 
        top += [skor[0]*c0 +  skor[1]*c1 +  skor[2]*c2 + skor[3]*c3 + skor[4]*c4]
    #print(top)
    arr = np.array(top)
    idx=arr.argsort()[-ozetCS:][::-1]
    #print(arr[idx])
    minE=min(arr[idx])
    res=[]
    sl=len(label)
    for i in range(0,cumleSayisi):
        if(i in idx):
            res+=[1]
        else:
            res+=[0]

    testY = labelBySentences[indeks]
    resY = res[-cumleSayisi:]
    #print(testY)
    #print(resY)
    return compare(resY,testY,indeks,yaz)

def baselineExp(indeks,interval,yaz):
    
    #ideal katsayıları bulmak için kullanılır
    trainIdx=[]
    for x in range(0, 130):
        if(x in range(indeks, indeks+interval)):
            trainIdx += [0]
        else:
            trainIdx += [1]
    #print(trainIdx)
    #trainX=np.array(trainX) #(features[:1220])
    c0l=[0.1,0.5,1,5]
    c1l=[0.1,0.5,1,5]
    c2l=[0.1,0.5,1,5]
    c3l=[0.1,0.5,1,5]
    c4l=[0.1,0.5,1]
    
    
    #ideal katsayılar
    c0l=[5] #5
    c1l=[0.1] #0.1
    c2l=[0.5] #5
    c3l=[0.1] #1,5
    c4l=[0.3] #0.1,1
    
    for c0 in c0l:
            for c1 in c1l:
                for c2 in c2l:
                    for c3 in c3l:
                        for c4 in c4l:
                            rc=0
                            pr=0
                            f=0
                            for i in range(indeks,indeks+interval):
                                [rec,pre,fv]=baselineCompExp(i,c0,c1,c2,c3,c4,yaz)
                                rc += rec/interval;
                                pr += pre/interval;
                                f += fv/interval;
                            print("Baseline: (Puanlama Katsayıları) " + "c0: " + str(c0) + "  c1: " + str(c1) + "  c2:" + str(c2) + "  c3:" + str(c3) + "  c4:" + str(c4) + " *** recall: " + str(rc) + "; prec: " + str(pr) + "; f: " + str(f) + "\n")
                            


# In[9]:


def myExpClf(clf,indeks,interval,yaz):    
    trainIdx=[]
    for x in range(0, 130):
        if(x in range(indeks, indeks+interval)):
            trainIdx += [0]
        else:
            trainIdx += [1]
    #print(trainIdx)
    trainX=[]
    trainY=[]
    for x in range(0, 130):
        if(trainIdx[x]==1):
            trainX+=featuresBySentences[x]
            trainY+=labelBySentences[x]
    trainX=np.array(trainX) #(features[:1220])
    y = np.array(label[:1220])


    y = np.array(trainY)
    #clf = svm.SVC(gamma=g,C=c,probability=True)
    clf.fit(trainX, y)
    rc=0
    pr=0
    f=0
    for i in range(indeks, indeks+interval):
        ozetCS = int(round(len(featuresBySentences[i])/8.15183))
        if(ozetCS==0):
            ozetCS=1
        testX =np.array(featuresBySentences[i])
        testY = labelBySentences[i]
        res = clf.predict(testX).tolist()
        resP = clf.predict_proba(testX).tolist()
        propList = []
        for j in resP:
            propList+=[j[1]]
        #print(testY)
        #print(res)
        #print(propList)
        arr = np.array(propList)
        idx=arr.argsort()[-ozetCS:][::-1]
        #print(idx)
        rs=[]
        for k in range(0,len(featuresBySentences[i])):
            if(k in idx):
                rs+=[1]
            else:
                rs+=[0]
        #print(rs)
        #print("***")
        [rec,pre,fv]=compare(rs,testY,i,yaz)
        rc += rec/interval;
        pr += pre/interval;
        f += fv/interval;
    #print("recall: " + str(rc) + "; prec: " + str(pr) + "; f: " + str(f))
    return [rc,pr,f]


# In[10]:


# Random Forest
def myExpRFCV(interval, multiParameter,yaz):
    k = int(interval/5)
    
    if(multiParameter==1):
        El=[3,20,50] #20
        Dl=[3,5,7] #5
        MSl=[20, 50, 70] #50
    else:
        El=[50] #20
        Dl=[7] #5
        MSl=[20] #50
    for c1 in El:
        for c2 in Dl:
            for c3 in MSl:
                rcRF=0
                prRF=0
                fRF=0
                for v in range(0,10):
                    for x in range(0,5):
                        rf = RandomForestClassifier(n_estimators=c1,max_depth=c2,min_samples_split=c3,min_samples_leaf=15)
                        [r,p,f]=myExpClf(rf,x*k,k,yaz)
                        rcRF+=r/50
                        prRF+=p/50
                        fRF+=f/50
                print("RF deneyi ~  nEst: " + str(c1) + "  maxD:" + str(c2) + "  minSS:" + str(c3) + " recallRF: " + str(rcRF) + "; precRF: " + str(prRF) + "; fRF: " + str(fRF) + "\n")

# SVM
def myExpSVMCV(interval, multiParameter,yaz):
    k = int(interval/5)
    if(multiParameter==1):
        CList=[1,10,100]
        GList=[1,10,100]
    else:
        CList=[1]
        GList=[1]
    for g in GList:
        for c in CList:
            rcSVM=0
            prSVM=0
            fSVM=0
            clf = svm.SVC(gamma=g,C=c,probability=True)
            for x in range(0,5):
                [r,p,f]=myExpClf(clf,x*k,k,yaz)
                rcSVM+=r/5
                prSVM+=p/5
                fSVM+=f/5
            print("SVM deneyi ~ gamma:" + str(g) + " C:" + str(c) + " recallSVM: " + str(rcSVM) + "; precSVM: " + str(prSVM) + "; fSVM: " + str(fSVM) + "\n")
            
def myExpLRCV(interval):
    k = int(interval/5)
    rcLR=0
    prLR=0
    fLR=0
    clf =  GaussianNB(var_smoothing=1e-5)
    for x in range(0,5):
        [r,p,f]=myExpClf(clf,x*k,k)
        rcLR+=r/5
        prLR+=p/5
        fLR+=f/5
    print("recallLR: " + str(rcLR) + "; precLR: " + str(prLR) + "; fLR: " + str(fLR))


# In[11]:


def Experiment():
    newsCount = 130
    outOfRange = newsCount+1

    #sistemimizde kullanılmak istenen öznitelik 1 yapılmalıdır
    anlatim=1
    tf=1 
    konum=1
    baslik=1
    varlik=1
    
    #sistemimizin öznitelikleri ve etiketler hazırlanır
    prepare(anlatim,tf,konum,baslik,varlik)
    
    #baselin parametreleri --> baseIdx: başlangıç indeksi, baseInt: deney yapılacak haber sayısı
    baseIdx=0
    baseInt=newsCount
    #baseline yöntem
    print("***Baseline Sonuçları***")
    baselineExp(baseIdx, baseInt,0)
    
    print("***Sistemimizin İdeal Sonuçları***")
    myExpRFCV(newsCount,0,outOfRange)
    myExpSVMCV(newsCount,0,0)
    
    print("***Öznitelik Testleri***")
    print("anlatim olmadan:")
    anlatim=0
    prepare(anlatim,tf,konum,baslik,varlik)
    myExpSVMCV(newsCount,0,outOfRange)
    print("tf olmadan:")
    anlatim=1
    tf=0
    prepare(anlatim,tf,konum,baslik,varlik)
    myExpSVMCV(newsCount,0,outOfRange)
    print("konum olmadan:")
    tf=1
    konum=0
    prepare(anlatim,tf,konum,baslik,varlik)
    myExpSVMCV(newsCount,0,outOfRange)
    print("baslik olmadan:")
    konum=1
    baslik=0
    prepare(anlatim,tf,konum,baslik,varlik)
    myExpSVMCV(newsCount,0,outOfRange)
    print("varlik olmadan:")
    baslik=1
    varlik=0
    prepare(anlatim,tf,konum,baslik,varlik)
    myExpSVMCV(newsCount,0,outOfRange)
    
    print("***Parametre Testleri***")
    myExpRFCV(newsCount,1,outOfRange)
    myExpSVMCV(newsCount,1,outOfRange)


# In[12]:
if __name__ == '__main__':
    #global haberler
    arg = sys.argv[1:]
    newsCount = 130
    outOfRange = newsCount+1
    
    #sistemimizde kullanılmak istenen öznitelik 1 yapılmalıdır
    anlatim=1
    tf=1 
    konum=1
    baslik=1
    varlik=1
    
    #sistemimizin öznitelikleri ve etiketler hazırlanır
    prepare(anlatim,tf,konum,baslik,varlik)
    
    #komut satırından 'ornek' girildiyse
    rand = random.randint(0,129)
    if(len(arg)!=0):
        if(str(arg[0])=='ornek'):
            print("HABER İNDEKSİ: " + str(rand))
            print("HHABER METNİ:\n")
            print(".".join(haberler[rand][2]))
            print("HABER AÇIKLAMASI(REFERANS ÖZET):\n")
            print(".".join(haberler[rand][1]))
            #baselin parametreleri --> baseIdx: başlangıç indeksi, baseInt: deney yapılacak haber sayısı
            baseIdx=0
            baseInt=newsCount
            #baseline yöntem
            print("\n ***BASELINE ÖZET VE DEĞERLENDİRME *** \n")
            baselineExp(baseIdx, baseInt,rand)
            
            print("\n ***SİSTEMİMİZİN İDEAL ÖZET VE DEĞERLENDİRME(BURADAKİ DEĞERLENDİRME GENELDİR; BU METNE ÖZGÜ DEĞİLDİR)*** \n")
            myExpSVMCV(newsCount,0,rand)
    
    else:
        #baselin parametreleri --> baseIdx: başlangıç indeksi, baseInt: deney yapılacak haber sayısı
        baseIdx=0
        baseInt=newsCount
        #baseline yöntem
        print("***Baseline Sonuçları***")
        baselineExp(baseIdx, baseInt,outOfRange)
        
        print("***Sistemimizin İdeal Sonuçları***")
        myExpRFCV(newsCount,0,outOfRange)
        myExpSVMCV(newsCount,0,outOfRange)
        
        print("***Öznitelik Testleri***")
        print("anlatim olmadan:")
        anlatim=0
        prepare(anlatim,tf,konum,baslik,varlik)
        myExpSVMCV(newsCount,0,outOfRange)
        print("tf olmadan:")
        anlatim=1
        tf=0
        prepare(anlatim,tf,konum,baslik,varlik)
        myExpSVMCV(newsCount,0,outOfRange)
        print("konum olmadan:")
        tf=1
        konum=0
        prepare(anlatim,tf,konum,baslik,varlik)
        myExpSVMCV(newsCount,0,outOfRange)
        print("baslik olmadan:")
        konum=1
        baslik=0
        prepare(anlatim,tf,konum,baslik,varlik)
        myExpSVMCV(newsCount,0,outOfRange)
        print("varlik olmadan:")
        baslik=1
        varlik=0
        prepare(anlatim,tf,konum,baslik,varlik)
        myExpSVMCV(newsCount,0,outOfRange)
        
        print("***Parametre Testleri***")
        myExpRFCV(newsCount,1,outOfRange)
        myExpSVMCV(newsCount,1,outOfRange)



# In[ ]:





# In[ ]:




