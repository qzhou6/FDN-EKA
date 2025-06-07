# from mymodel import tweet_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re
import pickle
import copy
import spacy

# 加载预训练的spaCy模型
nlp = spacy.load("en_core_web_sm")

def extract_nouns(sentence):
    """
    使用spaCy进行词性标注并提取名词
    :param sentence: 输入句子
    :return: 名词列表
    """
    # 处理句子
    doc = nlp(sentence)
    
    # 提取名词
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    
    return nouns









# # 加载预训练模型和分词器
# model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

# # 使用pipeline进行命名实体识别
# nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


def clean_entity(entity):
    # 使用正则表达式去除标点符号和多余的空格
    return re.sub(r'[^\w\s]', '', entity).strip()


def extract_entities(sentence):
    # 对输入句子进行命名实体识别
    ner_results = nlp(sentence)
    
    # 提取实体文本
    entities = [clean_entity(result['word']) for result in ner_results]
    
    return entities







def modified_dataset(data):
    all_entities = []
    for item in data:
        entities = extract_entities(item['tweet'])
        item['entities'] = entities
        # 将两个列表合并并去重
        all_entities_set = set(all_entities)
        entities_set = set(entities)

        # 将entities中的元素添加到all_entities中
        all_entities_set.update(entities_set)

        # 转换回列表
        all_entities = list(all_entities_set)
    return data,all_entities
        
   
   
       
    






# for item in modified_test_dataset:
#     print(item['tweet'])
#     print('=======================')
#     entities = extract_entities(item['tweet'])
#     item['entities'] = entities
#     # 将两个列表合并并去重
#     all_entities_set = set(all_entities)
#     entities_set = set(entities)

#     # 将entities中的元素添加到all_entities中
#     all_entities_set.update(entities_set)

#     # 转换回列表
#     all_entities = list(all_entities_set)
    
#     print(all_entities)
#     print(entities)
    
# with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/all_test_entities02.pkl', 'wb') as f:
#     pickle.dump(all_entities, f)
# print("all_entities 已保存到文件 all_test_entities02.pkl 中")

# with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/test_dataset_have_entities02.pkl', 'wb') as f:
#     pickle.dump(all_tweet_dataset, f)
# print("test_dataset 已保存到文件 test_dataset_have_entities02.pkl 中")


    
    
    
    