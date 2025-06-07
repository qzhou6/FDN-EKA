import torch
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import torch.nn as nn




# 定义线性层将特征映射到相同的维度
class FeatureMapper(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureMapper, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# 将这些线性层加入到模型中
class CLIPWithFeatureMapping(torch.nn.Module):
    def __init__(self, text_mapper, image_mapper):
        super(CLIPWithFeatureMapping, self).__init__()
        
        self.text_mapper = text_mapper
        self.image_mapper = image_mapper

    def forward(self, text_features, image_features):
        mapped_text_features = self.text_mapper(text_features)
        mapped_image_features = self.image_mapper(image_features)
        return mapped_text_features, mapped_image_features
    
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




class Idea02WeiBoDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): A dictionary containing 'label', 'original_post', and 'image' keys with corresponding data.
        """
        self.post_id = data['post_id']
        self.original_post = data['original_post']
        self.post_features = data['post_features']
        self.image_id = data['image_id']
        self.image_feature = data['image_feature']
        self.external_feature = data['external_feature']
        self.label = torch.tensor(data['label'], dtype=torch.float32)
        

    def __len__(self):
        # 返回数据集的长度
        return len(self.label)

    def __getitem__(self, idx):
        # 根据索引返回对应的数据
        post_id = self.post_id[idx]
        original_post = self.original_post[idx]
        post_features = self.post_features[idx]
        image_id = self.image_id[idx]
        image_feature = self.image_feature[idx]
        external_feature = self.external_feature[idx]
        label = self.label[idx]
        
        
        return  post_id,original_post,post_features, image_id,image_feature,external_feature,label

def compute_similarity_matrix(text_features, image_features):
    similarity_matrix = cosine_similarity(text_features, image_features)
    return similarity_matrix

    
def train_model(train_dataloader, test_dataloader, model, epochs=50, learning_rate=1e-5):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_f1 = 0  # 初始最佳 F1 分数
    best_model_state = None  # 存储最佳模型状态
    metrics = []  # 用于存储每个epoch的性能指标

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for post_id,original_post,post_features, image_id,image_feature,external_feature,label in progress_bar:
            # text_features = batch['text_features']
            # image_features = batch['image_features']
            post_features = post_features.squeeze(1)
            labels = label.float().to(post_features.device)

            optimizer.zero_grad()

            # 映射特征到相同的维度
            mapped_text_features, mapped_image_features = model(post_features, image_feature)

            # 计算图像和文本特征的余弦相似度
            cosine_sim = F.cosine_similarity(mapped_image_features.unsqueeze(1), mapped_text_features.unsqueeze(0), dim=2)
            match_scores = cosine_sim.diag()  # 假设匹配得分是对角线元素

            # 将得分转换为负值，使虚假新闻得分更低
            match_scores = -match_scores
            match_probabilities = torch.sigmoid(match_scores)  # Sigmoid 转换得分为概率

            # 使用labels作为targets
            loss = F.binary_cross_entropy(match_probabilities, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 记录预测和真实标签，用于计算准确率等指标
            predictions = (match_probabilities > 0.5).long()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 计算并存储性能指标
        average_loss = total_loss / len(train_dataloader)  # 计算当前 epoch 的平均损失
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')
        epoch_metrics = (average_loss, acc, precision, recall, f1)
        metrics.append(epoch_metrics)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Acc: {acc}, P: {precision}, R: {recall}, F1: {f1}")

        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0:
            print(f"Evaluating at epoch {epoch+1}")
            evaluate_model(test_dataloader, model)

        # 更新最佳 F1 分数和保存模型
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            print(f"New best F1 {best_f1} at epoch {epoch+1}")

        # # 每5个epoch保存一次当前epoch的信息，不一定是最佳模型
        # if (epoch + 1) % 5 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': average_loss,
        #         'accuracy': acc,
        #         'precision': precision,
        #         'recall': recall,
        #         'f1_score': f1
        #     }, f'checkpoint_epoch_{epoch+1}.pt')

    # 保存最佳 F1 模型
    if best_model_state:
        torch.save(best_model_state, f'/home/zhouqing/codes/COOLANT/weibo/idea02ModelFile/best_model_f1_{best_f1:.4f}.pt')

    return metrics

# 定义评估函数
def evaluate_model(dataloader, model):
    model.eval()  # 设置模型为评估模式
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # 在评估模式下，不跟踪梯度
        for post_id,original_post,post_features, image_id,image_feature,external_feature,label in dataloader:
            # text_features = batch['text_features']
            # image_features = batch['image_features']
            post_features = post_features.squeeze(1)
            labels = label.float().to(post_features.device)
            
            # 映射特征到相同的维度
            mapped_text_features, mapped_image_features = model(post_features, image_feature)
            
            # 计算图像和文本特征的余弦相似度
            cosine_sim = F.cosine_similarity(mapped_image_features.unsqueeze(1), mapped_text_features.unsqueeze(0), dim=2)
            match_scores = cosine_sim.diag()
            
            # 将得分转换为负值，使虚假新闻得分更低
            match_scores = -match_scores
            match_probabilities = torch.sigmoid(match_scores)
            predictions = (match_probabilities > 0.5).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    print(f"Evaluation - Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    return acc, precision, recall, f1


