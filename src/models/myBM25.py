import re,os
import numpy as np
from gensim import corpora
from gensim.summarization import bm25
from nltk.stem.porter import PorterStemmer  
from tqdm import tqdm
import pickle
p_stemmer = PorterStemmer()  




dataPath = '../../Willll/'  # Relative path of homework data


# r=root, d=directories, f = files

DocList = []
QueryList = []
DocData = []
QueryData = []
stopList = []

with open('./stopList.txt','r') as stopFP:
    tmp = stopFP.read()
    stopList = tmp.split('\n')

def is_not_stopword(myWord):
    for stop in stopList:
        if stop == myWord:
            return False
    return True

def articleParser(myPath):
    with open(myPath, 'r') as fp:
        docData = fp.read().replace('\n', '')
    data = re.sub(r'[0-9_!@#$%^&*()\[\]<>=/;\-"\',.:~]', '', docData).lower()
    return data

# read Query List
with open(dataPath+'test/query_list.txt', 'r') as fp:
    tmpLine = fp.readline()
    while tmpLine:
        tmpLine = tmpLine.strip('\n')
        if tmpLine != '':
            QueryList.append(tmpLine)
        tmpLine = fp.readline()

# Read query data
for eachQ in QueryList:
    QueryData.append(articleParser(dataPath+'test/query/%s'%eachQ))


for r, d, f in os.walk(dataPath+'doc'):
    for file in f:
        DocList.append(file)

#------------------------

for eachD in tqdm(DocList):
    DocData.append(articleParser(dataPath+'doc/'+eachD))


article_list =[]
DInd = 0
for eachD in tqdm(DocData):
    eachDList = eachD.split(' ')
    tmpDict = {}
    tmpDict['name'] = DocList[DInd]
    # 詞干提取
    stemmed_tokens = [p_stemmer.stem(i) for i in eachDList]  
    final_tokens = list(filter(is_not_stopword,stemmed_tokens))
    tmpDict['token'] = final_tokens
    article_list.append(tmpDict)
    DInd += 1

file = open(dataPath+'DocuStore.txt', 'wb')
pickle.dump(article_list, file)
file.close()

#------------------------
# article_list =[]
# with open(dataPath+'DocuStore.txt', 'rb') as file:
#     article_list =pickle.load(file)
#------------------------


article_data_list = [x['token'] for x in article_list]
article_name_list = [x['name'] for x in article_list]



total = open(dataPath+'bm25Pretrain.txt', 'w')
bmf = open(dataPath+'bm25result.txt', 'w')
bmf.write('Query,RetrievedDocuments\n')
for qIndex,qData in zip(QueryList,QueryData):
    bmf.write('%s,'%qIndex)
    query_stemmed = [p_stemmer.stem(i) for i in qData.split(' ')]  
    print('%s query_stemmed :'%(qIndex),query_stemmed )

    # bm25模型
    bm25Model = bm25.BM25(article_data_list)
    # 逆文件頻率
    # average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    scores = bm25Model.get_scores(query_stemmed)
    npScores = np.asarray(scores)
    rankList = npScores.argsort()[::-1]
    
    DL = ' '
    ranks = DL.join([article_name_list[x] for x in rankList[:100]])
    pretrain = DL.join([article_name_list[x] for x in rankList[:10000]])
    bmf.write('%s\n'%ranks)
    total.write('%s \n'%pretrain)

    # print(ranks)