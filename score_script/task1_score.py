from transformers import AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction 
from nltk.util import ngrams 
import json 


def bleu_compute(reference, hypothesis, weights=(1, 0, 0, 0)): 
    return corpus_bleu([[reference]], [hypothesis], weights=weights, smoothing_function=SmoothingFunction().method1) 


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def val_data_answer_read(data_path):
    answer_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        val_list = json.load(f) 
    
    for pair in val_list: 
        answer_list.append(pair['answer']['txt'])
    return answer_list 




if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 
    refer_data_path = 'val_task1.json'
    answer_list = val_data_answer_read(refer_data_path) 
    with open('result.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines() 
    print(len(lines))
    print(len(answer_list))
    
    '''
    sum = 0.0
    for i in range(len(lines)): 
        x = tokenizer.tokenize(answer_list[i]) 
        y = tokenizer.tokenize(lines[i]) 
        sum += bleu_compute(x, y, (0,0,0,1)) 
    print(sum / len(lines)) 
    '''

    total_x = []
    for i in range(len(lines)):
        x = tokenizer.tokenize(lines[i])  
        total_x += x 
    
    print(distinct_n_sentence_level(total_x, 2)) 

    



    '''
    #x = '你 好 厉害 啊 啊'
    #y = '你 真 的 厉害'
    x = tokenizer.tokenize(x) 
    print(x)
    print(distinct_n_sentence_level(x, 1))
    y = tokenizer.tokenize(y)
    print(y)
    print(bleu_compute(y, x))
    '''