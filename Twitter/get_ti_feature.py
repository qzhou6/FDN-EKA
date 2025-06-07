import pickle
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from common_function import convert_label,get_nouns_description,extract_resnet_features
from get_entities import extract_nouns
import torch.nn as nn
import os
from tqdm import tqdm



# 加载预处理后的图像数据
with open('/home/zhouqing/codes/COOLANT/twitter/data/part01/all_pro_image.pkl', 'rb') as f:
    all_pro_image = pickle.load(f)

# 读取tweet.txt文件
tweet_file = '/home/zhouqing/codes/COOLANT/twitter/data/text/tweets.txt'
tweets_df = pd.read_csv(tweet_file, sep='\t')







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



# 定义根据imageId获取图像向量的函数
def get_image_vector(image_ids):
    image_ids = image_ids.split(',')
    image_vectors = [all_pro_image[image_id] for image_id in image_ids if image_id in all_pro_image]
    if not image_vectors:
        print(f"没有找到对应的imageId: {image_ids}")
        return None
    image_vectors = np.stack(image_vectors)
    return image_vectors.mean(axis=0)

# 加载BERT模型和tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)
# 将模型设为评估模式
bert_model.eval()

image_folder = '/home/zhouqing/codes/COOLANT/twitter/data/all_image'
# 处理数据集
processed_data = []

for _, row in tqdm(tweets_df.iterrows(), total=len(tweets_df)):
    tweetId = row['tweetId']
    tweetText = row['tweetText']
    imageId = row['imageId']
    label = convert_label(row['label'])
    image_vector = get_image_vector(imageId)
    
    if image_vector is not None:
        # 提取文本特征
        img_features = extract_resnet_features(image_vector, model_name='resnet50')
        entities = extract_nouns(tweetText)
        nouns_description = get_nouns_description(row,entities)
        tweetText_feature = get_text_features(tweetText,tokenizer,bert_model)
        external_feature = get_aggregated_features(nouns_description, tokenizer, bert_model)
        
            
        # # 查找图像路径
        # image_path_jpg = os.path.join(image_folder, f"{imageId}.jpg")
        # image_path_png = os.path.join(image_folder, f"{imageId}.png")
        # image_path_png = os.path.join(image_folder, f"{imageId}.jpeg")
        
        # if os.path.exists(image_path_jpg):
        #     image_path = image_path_jpg
        # elif os.path.exists(image_path_png):
        #     image_path = image_path_png
        # else:
        #     print(f"No image found for imageId {imageId}")
        #     continue  # 跳过没有图像的行
        
            
        
        processed_data.append((tweetId, tweetText,tweetText_feature, imageId,img_features,external_feature,label))

# 将processed_data保存到文件
with open('/home/zhouqing/codes/COOLANT/idea02/data/only_external_feature.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("processed_data 已保存到文件 only_external_feature.pkl 中")



