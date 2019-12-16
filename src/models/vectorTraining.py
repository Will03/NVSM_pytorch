import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os



dataPath = '../../Willll/'  # Relative path of homework data


# r=root, d=directories, f = files

DocList = []
QueryList = []
DocData = []
QueryData = []

def articleParser(myPath):
    with open(myPath, 'r') as fp:
        docData = fp.read().replace('\n', '')
    return docData

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

for eachD in DocList:
    DocData.append(articleParser(dataPath+'doc/'+eachD))


# TF-IDF
max_df = 0.95        # Ignore words with high df. (Similar effect to stopword filtering)
min_df = 5           # Ignore words with low df.
smooth_idf = True    # Smooth idf weights by adding 1 to df.
sublinear_tf = True  # Replace tf with 1 + log(tf).

# Rocchio (Below is a param set called Ide Dec-Hi)
alpha = 1
beta = 0.75
gamma = 0.15
rel_count = 5   # Use top-5 relevant documents to update query vector.
nrel_count = 1  # Use only the most non-relevant document to update query vector.
iters = 5
print('start train')
# Build TF-IDF vectors of docs and queries
vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
doc_tfidfs = vectorizer.fit_transform(DocData).toarray()
query_vecs = vectorizer.transform(QueryData).toarray()
print('start count simi')
# Rank documents based on cosine similarity
cos_sim = cosine_similarity(query_vecs, doc_tfidfs)
rankings = np.flip(cos_sim.argsort(), axis=1)

print('start write file')
limit = 600
for query_name, ranking in zip(QueryList, rankings):
    ranked_docs=''
    index = 0
    for idx in ranking:
        if index >=600:
            break
        ranked_docs += '%s,'%DocList[idx]
    with open('../../Willll/%s.txt'%query_name, mode='w') as file:
        file.write('%s' % (ranked_docs))
