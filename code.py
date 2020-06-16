import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
import math
import random
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split

from collections import Counter
from scipy.sparse import csr_matrix


def sigmoid(z):
    return 1 /(1+np.exp(-z))

def hypothesis(data,index,theta):
    thetaTranX = 0
    for i in range(len(index)):
        thetaTranX += theta[index[i]]*data[i]
    return sigmoid(thetaTranX)

def cost(X,Y,theta):
    costVal = 0
    for i in range(X.shape[0]):
        data = test_data[i].data
        index = test_data[i].indices
        h = hypothesis(data,index,theta)
        if Y[i] == -1:
            tempY = 0
        else:
            tempY = Y[i]
        costVal += tempY*math.log(h)+(1-tempY)*math.log(1-h)
    costVal = costVal/X.shape[0]
    return costVal

def gradient(data,index,y,j,theta):
    if y == -1:
        tempY = 0
    else:
        tempY = y
    pos = 0
    for k in range(len(index)):
        if j==index[k]:
            pos = k
    return (tempY-hypothesis(data,index,theta))*data[pos]

def updateRule(x,y,alpha,theta):
    data = x.data
    index = x.indices
    for j in range(len(theta)):
        if j in index:
            theta[j] = theta[j] + alpha*gradient(data,index,y,j,theta)
    return theta

def predict(theta,num):
    countN,countP = 0,0
    filename = "prediction_1587138.txt"
    f= open(filename,"w+")
    for i in range(test_data.shape[0]):
        data = test_data[i].data
        index = test_data[i].indices
        if hypothesis(data,index,theta) >= 0.5:
            f.write('+1')
            countP += 1
        else:
            f.write('-1')
            countN += 1
        f.write('\n')
    print(f'prediction summary @ {num} --> countP = {countP}  countN = {countN}')

def predict_on_val(theta,num):
    a,b,c = 0,0,0
    correctP = 0
    for i in range(X_val.shape[0]):
        data = X_val[i].data
        index = X_val[i].indices
        if hypothesis(data,index,theta) >= 0.5:
            if Y_val[i] == 1:
                a += 1
                correctP += 1
            elif Y_val[i] == -1:
                c += 1
        else:
            if Y_val[i] == 1:
                b += 1
            else:
                correctP += 1
    f = (2*a)/(2*a+b+c)
    accuracy = correctP/X_val.shape[0]
    print(f'F-measure at {num} is {f}')
    print(f'accuracy at {num} is {accuracy}')
    return [accuracy,f]

def load_data(filename):
    with open(filename, "r") as fh:
        return fh.readlines()

def text_preprocessing(lines):
    words_to_remove ={'film':0,'movi':0,'charact':0,'actor':0,'actress':0,'man':0,'men':0,'woman':0,'women':0,'male':0,'female':0,'would':0,'even':0,'scene':0,'could':0,'think':0,'also':0,'thing':0,'seem':0,'get':0,'made':0,'mani':0,'someth':0, 'director':0,'girl':0,'though':0,'name':0,'cinema':0,'sam':0,'hey':0,'christoph':0,'henri':0,'bruce':0,'adam':0, 'johnni':0,'eddi':0,'kevin':0,'jackson':0,'still':0,'give':0} 
    sentences = [sent_tokenize(BeautifulSoup(l, "lxml").text) for l in lines]
    words=[]
    lmtzr = WordNetLemmatizer()
    ps = PorterStemmer()
    for s in sentences:
        words.append([word_tokenize(si.lower()) for si in s])
    words_list = []
    stop_words = stopwords.words('english')
    for word in words:
        temp =[]
        for w in word:
            for item in w:
                if  item.isalpha() and item not in stop_words and len(item)>2:
                    val = lmtzr.lemmatize(ps.stem(item))
                    if val not in words_to_remove:
                        temp.append(val)
        words_list.append(temp)
    print(f'len(words_list) : {len(words_list)}')
    return words_list

def build_matrix(docs):
    
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        

    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0
    n = 0
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return [mat,ncols,idx,nnz]

def build_matrix_test(docs,idx,nnz):
    
    nrows = len(docs)
    ncols = len(idx)
    
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0
    n = 0
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        
        count = 0
        for j,k in enumerate(keys):
            if k in idx:
                count += 1
                ind[j+n] = idx[k]
                val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + count
        n += count
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_l2normalize(mat, copy=False, **kargs):
  
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr

    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
    print('csr_l2normalize !!!!!!!!')     
    if copy is True:
        return mat

def csr_idf(mat, copy=False, **kargs):
    
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  
    
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
    print('finished csr_idf') 
    return df if copy is False else mat

def process_input_for_LR():
    lines_train = load_data('train.dat') 
    lines_test = load_data('test.dat')
    print("Read train data")

    #preprocessing the training and test data
    processed_review_list_train = text_preprocessing(lines_train)     
    print("completed pre processing train")
    processed_review_list_test = text_preprocessing(lines_test) 
    print("completed pre processing test")
    mat, ncols_train,vocab,nnz = build_matrix(processed_review_list_train)
    mat_test = build_matrix_test(processed_review_list_test,vocab,nnz)
    # print(f'ncols_train : {ncols_train}  ncols_test : {ncols_test}')
    csr_tf_idf_list_train  = csr_l2normalize(csr_idf(mat,copy=True),copy=True)
    csr_tf_idf_list_test = csr_l2normalize(csr_idf(mat_test,copy=True),copy=True)
    
    return [csr_tf_idf_list_train,csr_tf_idf_list_test,ncols_train]

def logistic_regression(ncols):
    theta = np.zeros(ncols)
    nRows = X.shape[0]
    alpha = 1.0
    i = 0
    # cost_arr, c  = [],0
    highest_accuracy = -1
    highest_accuracy_pos = -1
    highest_f = -1
    highest_f_pos = -1
    random.seed(0)
    while i<=2982:
        r = random.randint(0, 25000)
        theta = updateRule(X[(i+int(r))%nRows],Y[(i+int(r))%nRows],alpha,theta)
        
        i += 1
        if (i%445==0 and i%725==0) and alpha > 0.00001:  
            alpha = alpha/2
            print(f'updated alpha : {alpha}')
    predict(theta,i-1)
    cost_of_train = cost(X,Y,theta)
    cost_of_val = cost(X_val,Y_val,theta)
    print(f'cost_of_train {cost_of_train} cost_of_val {cost_of_val}')
    print(f'varience of the model = {cost_of_train-cost_of_val}')



X_data ,test_data,ncols = process_input_for_LR()
Y_data = np.loadtxt('train.labels')
X, X_val,Y, Y_val = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 0)
logistic_regression(ncols)

 


