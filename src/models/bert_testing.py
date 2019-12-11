import pickle
from pathlib import Path

import torch,os,re
import torch.optim as optim
from torch.utils.data import DataLoader

from utils          import create_dataset, create_pytorch_datasets, create_query_dataset, \
                           evaluate_queries
from train_model    import train
from evaluate_model import evaluate, generate_eval
from evaluate_bert  import evaluate_queries_bert

from nvsm_bert import NVSMBERT, loss_function

from transformers import BertTokenizer
#from transformers.optimization import BertAdam
#from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

def articleParser(myPath):
    with open(myPath, 'r') as fp:
        docData = fp.read().replace('\n', '')
    data = re.sub(r'[0-9_!@#$%^&*()\[\]<>=/;\-"\',.:~]', '', docData).lower()
    return data

def load_data(data_folder, testing_query_folder,pretrained_model):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    #with open(data_folder / 'tokenized_docs.pkl', 'rb') as tok_docs_file:
    #     docs = pickle.load(tok_docs_file)
    docFiles = os.listdir(data_folder)
    docs = []
    for docFileName in docFiles:
        tmpDict = {}
        docWordList = articleParser(data_folder / docFileName).split()
        tmpDict['name'] = docFileName
        tmpDict['tokens'] = docWordList
        docs.append(tmpDict)
    
    queryFiles = os.listdir(testing_query_folder)
    query = []
    for queryFileName in queryFiles:
        tmpQuery = {}
        queryWords = articleParser(testing_query_folder / queryFileName)
        tmpQuery['name'] = queryFileName
        tmpQuery['tokens'] = queryWords
        query.append(tmpQuery)
    return docs, query,tokenizer

def create_dataset(tok_docs, tokenizer, n):
    '''
    Creates the dataset by extracting n-grams from the documents using a
    rolling window.
    '''
    n_grams      = []
    document_ids = []
    unk_tok_id   = tokenizer.vocab['[UNK]']
    cls_tok_id   = tokenizer.vocab['[CLS]']
    sep_tok_id   = tokenizer.vocab['[SEP]']
    for i, doc in enumerate(tok_docs):
        doc_tok_ids = tokenizer.encode(' '.join(doc))[1:-1]
        for n_gram in [doc_tok_ids[i : i + n] for i in range(len(doc) - n)]:
            if all(tok == unk_tok_id for tok in n_gram):
                continue
            n_grams.append([cls_tok_id] + n_gram + [sep_tok_id])
            document_ids.append(i)

    return n_grams, document_ids

def main():
    mypath = r'/home/connlab/108IR/will/final/NVSM_pytorch/'
    #mypath = r'C:/Users/willll/Desktop/WIillll/IRCLass/Final/NVSM_pytorch'
    pretrained_model      = 'bert-base-uncased'
    glove_path            = Path(mypath + '/glove')
    model_folder          = Path(mypath + '/models')
   # data_folder           = Path(mypath + '/data/processed')
    data_folder           = Path(mypath + '/Willll/fakedoc')
    testing_query_folder  = Path(mypath + '/Willll/test/query')
    model_path            = model_folder / 'nvsm_bert.pt'
    batch_size            = 140 # for 150, 8053 / 8113MB GPU memory, to tweak
    epochs                = 3
    docs, queries ,tokenizer       = load_data(
        data_folder,
        testing_query_folder,
        pretrained_model
    )
    # docs = docs[:20]
    doc_names             = [doc['name'] for doc in docs]
    n_grams, document_ids = create_dataset(
        tok_docs  = [doc['tokens'] for doc in docs],
        tokenizer = tokenizer,
        n         = 10
    )

    print('N-grams number', len(n_grams))
    k_values              = [1, 3, 5, 10]
    (train_data,
     eval_data,
     eval_train_data)     = create_pytorch_datasets(n_grams, document_ids)
    print('Train dataset size', len(train_data))
    print('Eval dataset size', len(eval_data))
    print('Eval (training) dataset size', len(eval_train_data))


    eval_loader           = DataLoader(eval_data, batch_size = batch_size, shuffle = False)
    eval_train_loader     = DataLoader(eval_train_data, batch_size = batch_size, shuffle = False)
    device                = torch.device('cuda')
    lamb                  = 1e-3

    nvsm                  = NVSMBERT(
        pretrained_model  = pretrained_model,
        n_doc             = len(doc_names),
        dim_doc_emb       = 20,
        neg_sampling_rate = 10,
    ).to(device)
    #torch.save(nvsm.state_dict(), model_path)
    nvsm.load_state_dict(torch.load(model_path))
    nvsm.eval()
    recall_at_ks = evaluate(
        nvsm          = nvsm,
        device        = device,
        eval_loader   = eval_loader,
        recalls       = k_values,
        loss_function = loss_function,
    )
    print(generate_eval(k_values, recall_at_ks))
    queries_text             = [query['tokens'] for query in queries]
    queries_name             = [query['name'] for query in queries]
#    queries_text          = [
#        'violence king louis decapitated',
#        'domain language translate',
#        'governement robespierre',
#        'perfect imperfect information',
#        'ontology translation',
#        'high levels of political violence',
#        'state education system which promotes civic values',
#        'political struggles',
#        'Almost all future revolutionary movements looked back to the Revolution as their predecessor',
#        'Habermas argued that the dominant cultural model in 17th century France was a "representational" culture',
#        'mathematical model winning strategy',
#        'solutions for two-person zero-sum games',
#        'cooperative coalitions bargaining',
#        'eigenvalue',
#        'graph, dimension and components',
#        'inner product vertex'
#    ]
    
    

    
    evaluation_results = evaluate_queries_bert(
        nvsm,
        queries_text,
        doc_names,
        tokenizer,
        batch_size,
        device
    )
    for query_name,query_text, doc_idx in zip(queries_name,queries_text, evaluation_results):
        print(f'{query_name} {query_text:35} -> {doc_names[doc_idx]}')

if __name__ == '__main__':
    main()
