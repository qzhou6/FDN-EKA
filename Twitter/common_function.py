import re 
import requests
from tqdm import tqdm   
# from link_wiki import entity_description_dict
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from get_entities import extract_nouns
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity



def filter_words(word_list):
    """
    过滤列表中的单词，保留只包含英文字母和数字组合的项，过滤掉网址、带标点符号的词等。
    :param word_list: 输入的单词列表
    :return: 过滤后的单词列表
    """
    # 正则表达式模式：匹配只包含英文字母和数字组合的项
    pattern = re.compile(r'^[a-zA-Z0-9]+$')
    
    # 过滤列表
    filtered_list = [word for word in word_list if pattern.match(word)]
    
    return filtered_list




def get_wikidata_description(entity):
    # Step 1: Get the Wikidata entity ID from Wikipedia
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": entity,
        "prop": "pageprops"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    pages = data.get("query", {}).get("pages", {})
    wikidata_id = None
    for page_id, page_content in pages.items():
        if page_id != "-1":
            wikidata_id = page_content.get("pageprops", {}).get("wikibase_item")
            break

    if not wikidata_id:
        return "Wikidata entity not found."

    # Step 2: Get the entity description from Wikidata
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": wikidata_id,
        "props": "descriptions",
        "languages": "en"
    }
    response = requests.get(url, params=params)
    data = response.json()

    descriptions = data.get("entities", {}).get(wikidata_id, {}).get("descriptions", {})
    description = descriptions.get("en", {}).get("value", "No description available.")

    return description


# def create_entity_description_dict(filter_nouns):
#     """
#     遍历filter_nouns列表，获取每个名词的描述，并存储在字典中。
#     如果没有找到描述，则存储为null。
#     :param filter_nouns: 名词列表
#     :return: 名词描述字典
#     """
#     entity_description_dict = {}
    
#     for noun in tqdm(filter_nouns, desc="Processing nouns"):
#         description = get_wikidata_description(noun)
#         if description == "Wikidata entity not found.":
#             entity_description_dict[noun] = None
#         else:
#             entity_description_dict[noun] = description
    
#     return entity_description_dict


def split_list_in_half(input_list):
    """
    将输入列表对半拆分成两个列表
    :param input_list: 输入的列表
    :return: 两个拆分后的列表
    """
    # 计算列表长度的一半
    half_length = len(input_list) // 2
    
    # 使用切片操作拆分列表
    list1 = input_list[:half_length]
    list2 = input_list[half_length:]
    
    return list1, list2


def merge_dictionaries(dict1, dict2):
    """
    将两个字典合并成一个字典
    :param dict1: 第一个字典
    :param dict2: 第二个字典
    :return: 合并后的字典
    """
    merged_dict = dict1.copy()  # 创建第一个字典的副本
    merged_dict.update(dict2)   # 使用update方法将第二个字典合并到副本中
    return merged_dict


# # 定义生成名词描述列表的函数
# def get_nouns_description(row):
#     entities = row['entities']
#     if not entities:  # 如果 entities 为空
#         return [row['tweet']]
    
#     nouns_description = []
#     for noun in entities:
#         description = entity_description_dict.get(noun)
#         if description is None:
#             description = row['tweet']
#         nouns_description.append(description)
    
#     # 检查是否所有描述都是 tweet 本身（表示没有找到描述）
#     if all(desc == row['tweet'] for desc in nouns_description):
#         return [row['tweet']]
    
#     return nouns_description



# def get_nouns_description(row,entities):
    
#     if not entities:  # 如果 entities 为空
#         return [row['tweetText']]
    
#     nouns_description = []
#     for noun in entities:
#         description = entity_description_dict.get(noun)
#         if description is None:
#             description = row['tweetText']
#         nouns_description.append(description)
    
#     # 检查是否所有描述都是 tweet 本身（表示没有找到描述）
#     if all(desc == row['tweetText'] for desc in nouns_description):
#         return [row['tweetText']]
    
#     return nouns_description





def extract_resnet_features(image_vector, model_name='resnet50', pretrained=True):
    """
    提取经过预训练的 ResNet 模型的特征。

    参数:
    - image_vector (torch.Tensor): 预处理后的图像向量，形状为 [3, 224, 224]。
    - model_name (str): ResNet 模型的名称，可以是 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'。
    - pretrained (bool): 是否使用预训练模型。

    返回:
    - features (torch.Tensor): 提取的特征向量。
    """
    
    # 如果 image_vector 是 numpy.ndarray，则转换为 torch.Tensor
    if isinstance(image_vector, np.ndarray):
        image_vector = torch.tensor(image_vector)
    
    # 加载预训练的 ResNet 模型
    if model_name == 'resnet18':
        resnet = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        resnet = models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        resnet = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        resnet = models.resnet101(pretrained=pretrained)
    elif model_name == 'resnet152':
        resnet = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError("Invalid model_name. Choose from 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'.")

    # 将模型设为评估模式
    resnet.eval()

    # 添加批量维度，变成 [1, 3, 224, 224]
    image_vector = image_vector.unsqueeze(0)

    # 前向传播，提取特征
    with torch.no_grad():
        features = resnet(image_vector)

    # 移除批量维度，特征维度应为 [1000]
    features = features.squeeze()

    return features


# 定义label转换函数
def convert_label(label):
    if label in ['fake', 'humor']:
        return 1
    elif label == 'real':
        return 0
    else:
        raise ValueError(f"未知的label: {label}")
    


def get_text_features(text, tokenizer, model):
    """
    获取单个文本的 BERT 特征。

    参数:
    - text (str): 文本信息
    - tokenizer (BertTokenizer): BERT 分词器
    - model (BertModel): BERT 模型

    返回:
    - features (torch.Tensor): 文本的特征向量
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取最后一层隐藏状态的均值
    features = outputs.last_hidden_state.mean(dim=1)
    return features

def get_aggregated_features(nouns_description, tokenizer, model):
    """
    获取 nouns_description 列表中所有文本的平均 BERT 特征。

    参数:
    - nouns_description (list): 文本信息列表
    - tokenizer (BertTokenizer): BERT 分词器
    - model (BertModel): BERT 模型

    返回:
    - aggregated_features (torch.Tensor): 平均特征向量
    """
    features_list = [get_text_features(text, tokenizer, model) for text in nouns_description]
    # 将所有特征向量堆叠在一起
    features_tensor = torch.stack(features_list)
    # 取均值
    aggregated_features = features_tensor.mean(dim=0)
    return aggregated_features

def compute_similarity_matrix(text_features, image_features):
    similarity_matrix = cosine_similarity(text_features, image_features)
    return similarity_matrix

