import torch
import pickle
from torch.utils.data import Dataset,DataLoader,random_split
import torch.nn as nn
import torch.nn.functional as F





# 将这些线性层加入到模型中
class CLIPWithFeatureMapping(torch.nn.Module):
    def __init__(self, clip_model, text_mapper, image_mapper):
        super(CLIPWithFeatureMapping, self).__init__()
        self.clip_model = clip_model
        self.text_mapper = text_mapper
        self.image_mapper = image_mapper

    def forward(self, text_features, image_features):
        mapped_text_features = self.text_mapper(text_features)
        mapped_image_features = self.image_mapper(image_features)
        return mapped_text_features, mapped_image_features

# 定义线性层将特征映射到相同的维度
class FeatureMapper(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



class Tweet_feature_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取数据项
        data_item = self.data[idx]
        
        # 检查数据项是否包含 None
        if any(item is None for item in data_item):
            # 返回一个默认值或处理方法
            return None

        tweetId, tweetText, tweetText_feature, imageId, img_features, external_feature, label = data_item        
        tmp = {
            'tweetId': tweetId,
            'tweetText':tweetText,
            'tweetText_feature': tweetText_feature,
            'imageId': imageId,
            'img_features': img_features,
            'external_feature' : external_feature,
            'label': label
        }
        return tmp
        
# 自定义 collate 函数
def custom_collate_fn(batch):
    # 过滤掉 None 数据
    batch = [item for item in batch if item is not None]
    
    tweetIds = [item['tweetId'] for item in batch]
    tweetTexts = [item['tweetText'] for item in batch]
    tweetText_features = torch.stack([item['tweetText_feature'] for item in batch])
    imageIds = [item['imageId'] for item in batch]
    img_features = torch.stack([item['img_features'] for item in batch])
    external_features = torch.stack([item['external_feature'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])

    return tweetIds, tweetTexts, tweetText_features, imageIds, img_features, external_features, labels
        


# 从文件中加载processed_data
with open('/home/zhouqing/codes/COOLANT/idea02/data/only_external_feature.pkl', 'rb') as f:
    processed_data = pickle.load(f)


# 创建数据集实例
tweet_dataset = Tweet_feature_Dataset(processed_data)






# 划分训练集和测试集
test_size = int(0.2 * len(tweet_dataset))
train_size = len(tweet_dataset) - test_size
train_dataset, test_dataset = random_split(tweet_dataset, [train_size, test_size])
print(len(train_dataset),len(test_dataset))
from collections import Counter

# 统计 train_dataset 中标签为 0 和 1 的数量
train_labels = [data['label'] for data in train_dataset]
train_label_counts = Counter(train_labels)

# 统计 test_dataset 中标签为 0 和 1 的数量
test_labels = [data['label'] for data in test_dataset]
test_label_counts = Counter(test_labels)

# 打印标签为 0 和 1 的数量
print(f"Train dataset - Label 0: {train_label_counts.get(0, 0)}, Label 1: {train_label_counts.get(1, 0)}")
print(f"Test dataset - Label 0: {test_label_counts.get(0, 0)}, Label 1: {test_label_counts.get(1, 0)}")


# 创建DataLoader
batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

# # 获取一个批次的数据
# for tweetIds, tweetTexts, tweetText_features, imageIds, img_features, external_features, labels in train_dataloader:
#     print("tweetIds:", tweetIds)
#     print("tweetTexts:", tweetTexts)
#     print("tweetText_features shape:", tweetText_features.shape)
#     print("imageIds:", imageIds)
#     print("img_features shape:", img_features.shape)
#     print("external_features shape:", external_features.shape)
#     print("labels:", labels)
#     break  # 这里只是示例，获取一个批次的数据即可


class CustomClassifier(nn.Module):
    def __init__(self, img_feature_dim, ext_feature_dim, txt_feature_dim, fusion_dim, num_classes=2):
        super(CustomClassifier, self).__init__()
        self.img_feature_dim = img_feature_dim
        self.ext_feature_dim = ext_feature_dim
        self.txt_feature_dim = txt_feature_dim
        self.fusion_dim = fusion_dim

        # 定义输入特征的全连接层
        self.img_fc = nn.Linear(img_feature_dim, fusion_dim)
        self.ext_fc = nn.Linear(ext_feature_dim, fusion_dim)
        self.txt_fc = nn.Linear(txt_feature_dim, fusion_dim)
        self.ext_txt_fc = nn.Linear(ext_feature_dim + txt_feature_dim, fusion_dim)

        # 定义最后的分类层
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, img_features, ext_features, txt_features):
        # Reshape external and text features to [batch_size, feature_dim]
        ext_features = ext_features.view(ext_features.size(0), -1)
        txt_features = txt_features.view(txt_features.size(0), -1)

        # 特征变换
        img_f = self.img_fc(img_features)
        ext_f = self.ext_fc(ext_features)
        txt_f = self.txt_fc(txt_features)
        ext_txt_f = self.ext_txt_fc(torch.cat((ext_features, txt_features), dim=-1))

        # 计算 KL 散度
        kl_img_ext = F.kl_div(F.log_softmax(img_f, dim=-1), F.softmax(ext_f, dim=-1), reduction='batchmean')
        kl_img_txt = F.kl_div(F.log_softmax(img_f, dim=-1), F.softmax(txt_f, dim=-1), reduction='batchmean')
        kl_img_ext_txt = F.kl_div(F.log_softmax(img_f, dim=-1), F.softmax(ext_txt_f, dim=-1), reduction='batchmean')

        # 计算权重
        kl_sum = kl_img_ext + kl_img_txt + kl_img_ext_txt
        weight_ext = kl_img_ext / kl_sum
        weight_txt = kl_img_txt / kl_sum
        weight_ext_txt = kl_img_ext_txt / kl_sum

        # 特征融合
        fusion_features = img_f * (1 - (weight_ext + weight_txt + weight_ext_txt)) + \
                          ext_f * weight_ext + txt_f * weight_txt + ext_txt_f * weight_ext_txt

        # 分类
        output = self.classifier(fusion_features)

        return output
    
    
class CustomClassifiernoExternal(nn.Module):
    def __init__(self, img_feature_dim, txt_feature_dim, fusion_dim, num_classes=2):
        super(CustomClassifiernoExternal, self).__init__()
        self.img_feature_dim = img_feature_dim
        self.txt_feature_dim = txt_feature_dim
        self.fusion_dim = fusion_dim

        # 定义输入特征的全连接层
        self.img_fc = nn.Linear(img_feature_dim, fusion_dim)
        self.txt_fc = nn.Linear(txt_feature_dim, fusion_dim)

        # 定义最后的分类层
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, img_features, txt_features):
        # Reshape text features to [batch_size, feature_dim]
        txt_features = txt_features.view(txt_features.size(0), -1)

        # 特征变换
        img_f = self.img_fc(img_features)
        txt_f = self.txt_fc(txt_features)

        # 特征融合
        fusion_features = img_f + txt_f

        # 分类
        output = self.classifier(fusion_features)

        return output
    

 
 # 定义计算总变差距离（TVD）的函数
def total_variation_distance(p, q):
    # 将张量转换为概率分布（使用softmax）
    p_prob = torch.softmax(p, dim=-1)
    q_prob = torch.softmax(q, dim=-1)
    # 计算总变差距离
    tvd = 0.5 * torch.sum(torch.abs(p_prob - q_prob), dim=-1)
    return tvd.mean()
 
 
 # 定义计算欧氏距离的函数
def euclidean_distance(x, y):
    # 计算每个样本之间的差的平方和，然后取平方根
    distance = torch.sqrt(torch.sum((x - y) ** 2, dim=-1))
    return distance.mean()
 
 
    
class CustomClassifiertvd(nn.Module):
    def __init__(self, img_feature_dim, ext_feature_dim, txt_feature_dim, fusion_dim, num_classes=2):
        super(CustomClassifiertvd, self).__init__()
        self.img_feature_dim = img_feature_dim
        self.ext_feature_dim = ext_feature_dim
        self.txt_feature_dim = txt_feature_dim
        self.fusion_dim = fusion_dim

        # 定义输入特征的全连接层
        self.img_fc = nn.Linear(img_feature_dim, fusion_dim)
        self.ext_fc = nn.Linear(ext_feature_dim, fusion_dim)
        self.txt_fc = nn.Linear(txt_feature_dim, fusion_dim)
        self.ext_txt_fc = nn.Linear(ext_feature_dim + txt_feature_dim, fusion_dim)

        # 定义最后的分类层
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, img_features, ext_features, txt_features):
        # Reshape external and text features to [batch_size, feature_dim]
        ext_features = ext_features.view(ext_features.size(0), -1)
        txt_features = txt_features.view(txt_features.size(0), -1)

        # 特征变换
        img_f = self.img_fc(img_features)
        ext_f = self.ext_fc(ext_features)
        txt_f = self.txt_fc(txt_features)
        ext_txt_f = self.ext_txt_fc(torch.cat((ext_features, txt_features), dim=-1))

        # 计算总变差距离
        tvd_img_ext = total_variation_distance(img_f, ext_f)
        tvd_img_txt = total_variation_distance(img_f, txt_f)
        tvd_img_ext_txt = total_variation_distance(img_f, ext_txt_f)

        # 计算权重
        tvd_sum = tvd_img_ext + tvd_img_txt + tvd_img_ext_txt
        weight_ext = tvd_img_ext / tvd_sum
        weight_txt = tvd_img_txt / tvd_sum
        weight_ext_txt = tvd_img_ext_txt / tvd_sum

        # 特征融合
        fusion_features = img_f * (1 - (weight_ext + weight_txt + weight_ext_txt)) + \
                          ext_f * weight_ext + txt_f * weight_txt + ext_txt_f * weight_ext_txt

        # 分类
        output = self.classifier(fusion_features)

        return output
    
    
    
class CustomClassifiered(nn.Module):
    def __init__(self, img_feature_dim, ext_feature_dim, txt_feature_dim, fusion_dim, num_classes=2):
        super(CustomClassifiered, self).__init__()
        self.img_feature_dim = img_feature_dim
        self.ext_feature_dim = ext_feature_dim
        self.txt_feature_dim = txt_feature_dim
        self.fusion_dim = fusion_dim

        # 定义输入特征的全连接层
        self.img_fc = nn.Linear(img_feature_dim, fusion_dim)
        self.ext_fc = nn.Linear(ext_feature_dim, fusion_dim)
        self.txt_fc = nn.Linear(txt_feature_dim, fusion_dim)
        self.ext_txt_fc = nn.Linear(ext_feature_dim + txt_feature_dim, fusion_dim)

        # 定义最后的分类层
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, img_features, ext_features, txt_features):
        # Reshape external and text features to [batch_size, feature_dim]
        ext_features = ext_features.view(ext_features.size(0), -1)
        txt_features = txt_features.view(txt_features.size(0), -1)

        # 特征变换
        img_f = self.img_fc(img_features)
        ext_f = self.ext_fc(ext_features)
        txt_f = self.txt_fc(txt_features)
        ext_txt_f = self.ext_txt_fc(torch.cat((ext_features, txt_features), dim=-1))

        # 计算欧氏距离
        ed_img_ext = euclidean_distance(img_f, ext_f)
        ed_img_txt = euclidean_distance(img_f, txt_f)
        ed_img_ext_txt = euclidean_distance(img_f, ext_txt_f)

        # 计算权重
        ed_sum = ed_img_ext + ed_img_txt + ed_img_ext_txt
        weight_ext = ed_img_ext / ed_sum
        weight_txt = ed_img_txt / ed_sum
        weight_ext_txt = ed_img_ext_txt / ed_sum

        # 特征融合
        fusion_features = img_f * (1 - (weight_ext + weight_txt + weight_ext_txt)) + \
                          ext_f * weight_ext + txt_f * weight_txt + ext_txt_f * weight_ext_txt

        # 分类
        output = self.classifier(fusion_features)

        return output