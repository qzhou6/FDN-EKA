import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from idea02Function import compute_similarity_matrix
from idea02Dataset import train_dataloader,test_dataloader
from idea02Function import FeatureMapper,CLIPWithFeatureMapping,CustomClassifier
import numpy as np



def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None)
    return precision, recall, f1




# 初始化线性层，假设我们将特征映射到512维
text_feature_mapper = FeatureMapper(768, 512)  # 768是BERT的默认输出维度
image_feature_mapper = FeatureMapper(1000, 512)  # 1000是ResNet的默认输出维度

# 实例化模型
clip_with_mapping = CLIPWithFeatureMapping( text_feature_mapper, image_feature_mapper)
# 加载模型
model_path = "/home/zhouqing/codes/COOLANT/weibo/idea02ModelFile/best_model_f1_0.9118.pt"
checkpoint = torch.load(model_path)

clip_with_mapping.load_state_dict(checkpoint['model_state_dict'])
# text_feature_mapper.load_state_dict(checkpoint['optimizer_state_dict'])







        
# 模型和训练参数
img_feature_dim = 512
ext_feature_dim = 512
txt_feature_dim = 512
fusion_dim = 1024
num_classes = 2
num_epochs = 100
learning_rate = 0.001

# 创建模型、损失函数和优化器
model = CustomClassifier(img_feature_dim, ext_feature_dim, txt_feature_dim, fusion_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
clip_with_mapping = clip_with_mapping.to(device)
clip_with_mapping.eval()  # 设置为评估模式

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_labels = []
    all_preds = []

    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    for post_id,original_post,post_features, image_id,image_feature,external_feature,label in progress_bar:
        # 将数据移动到 GPU（如果有的话）
        img_features = image_feature.to(device)
        external_features = external_feature.to(device)
        external_features = external_features.squeeze(1)
        post_features = post_features.to(device)
        post_features = post_features.squeeze(1)
        label = label.long().to(device)
        
        fused_features = external_features * post_features
        mapped_text_features, mapped_image_features = clip_with_mapping(post_features, img_features)
        mapped_fused_features, mapped_image_features02 = clip_with_mapping(fused_features, img_features)
        mapped_ex_features, mapped_image_features03 = clip_with_mapping(external_features, img_features)
        
        mapped_text_features_orin = mapped_text_features
        mapped_image_features_orin = mapped_image_features
        mapped_ex_features_ori = mapped_ex_features
        
        batch_size = mapped_text_features.size(0)
        mapped_text_features = mapped_text_features.detach().cpu().numpy().reshape(batch_size, -1)
        mapped_image_features = mapped_image_features.detach().cpu().numpy()
        mapped_fused_features = mapped_fused_features.detach().cpu().numpy().reshape(batch_size, -1)
        mapped_image_features02 = mapped_image_features02.detach().cpu().numpy()
        mapped_ex_features = mapped_ex_features.detach().cpu().numpy().reshape(batch_size, -1)
        mapped_image_features03 = mapped_image_features03.detach().cpu().numpy()
        
        similarity_matrix = compute_similarity_matrix(mapped_text_features, mapped_image_features)
        fused_similarity_matrix = compute_similarity_matrix(mapped_ex_features, mapped_image_features03)
        

        # Using model outputs for loss computation
        outputs = model(mapped_image_features_orin, mapped_ex_features_ori, mapped_text_features_orin)
        pred_labels = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, label)  # 'outputs' are logits, 'labels' are target class indices

        # 清零梯度
        optimizer.zero_grad()

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * label.size(0)
        total_samples += label.size(0)
        all_labels.extend(label.cpu().numpy())
        all_preds.extend(pred_labels.cpu().numpy())

        # 更新进度条的描述
        progress_bar.set_postfix({"loss": running_loss / total_samples})



    precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    precision_real, precision_fake = precision
    recall_real, recall_fake = recall
    f1_real, f1_fake = f1
    print(f'Train Precision (Real): {precision_real:.4f}, Recall (Real): {recall_real:.4f}, F1 Score (Real): {f1_real:.4f}')
    print(f'Train Precision (Fake): {precision_fake:.4f}, Recall (Fake): {recall_fake:.4f}, F1 Score (Fake): {f1_fake:.4f}')

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    # pall_predictions
    if (epoch + 1) % 5 == 0:
        model.eval()
        test_labels = []
        test_preds = []

        with torch.no_grad():
            for post_id,original_post,post_features, image_id,image_feature,external_feature,label in test_dataloader:
                image_feature = image_feature.to(device)
                external_feature = external_feature.to(device)
                external_feature = external_feature.squeeze(1)
                post_features = post_features.to(device)
                post_features = post_features.squeeze(1)
                label = label.long().to(device)
                
                fused_features = external_feature * post_features
                mapped_text_features, mapped_image_features = clip_with_mapping(post_features, image_feature)
                mapped_fused_features, mapped_image_features02 = clip_with_mapping(fused_features, image_feature)
                mapped_ex_features, mapped_image_features03 = clip_with_mapping(external_feature, image_feature)
                
                mapped_text_features_orin = mapped_text_features
                mapped_image_features_orin = mapped_image_features
                mapped_ex_features_ori = mapped_ex_features
                
                batch_size = mapped_text_features.size(0)
                mapped_text_features = mapped_text_features.detach().cpu().numpy().reshape(batch_size, -1)
                mapped_image_features = mapped_image_features.detach().cpu().numpy()
                mapped_fused_features = mapped_fused_features.detach().cpu().numpy().reshape(batch_size, -1)
                mapped_image_features02 = mapped_image_features02.detach().cpu().numpy()
                mapped_ex_features = mapped_ex_features.detach().cpu().numpy().reshape(batch_size, -1)
                mapped_image_features03 = mapped_image_features03.detach().cpu().numpy()
                
                similarity_matrix = compute_similarity_matrix(mapped_text_features, mapped_image_features)
                fused_similarity_matrix = compute_similarity_matrix(mapped_ex_features, mapped_image_features03)
                manual_predictions = []
                
                outputs = model(mapped_image_features_orin, mapped_ex_features_ori, mapped_text_features_orin)
                pred_labels = torch.argmax(outputs, dim=1)

                test_labels.extend(label.cpu().numpy())
                test_preds.extend(pred_labels.cpu().numpy())
                
                

                
        # 计算测试指标
        precision, recall, f1 = calculate_metrics(test_labels, test_preds)
        precision_real, precision_fake = precision
        recall_real, recall_fake = recall
        f1_real, f1_fake = f1
        # 计算准确率
        correct_predictions = (np.array(all_labels) == np.array(all_preds)).sum()
        accuracy = correct_predictions / len(all_labels)
        print(f'Test Accuracy: {accuracy:.4f}')

        print(f'Test Precision (Real): {precision_real:.4f}, Recall (Real): {recall_real:.4f}, F1 Score (Real): {f1_real:.4f}')
        print(f'Test Precision (Fake): {precision_fake:.4f}, Recall (Fake): {recall_fake:.4f}, F1 Score (Fake): {f1_fake:.4f}')

print('Finished Training')
        

        

