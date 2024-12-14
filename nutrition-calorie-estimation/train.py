import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from models import MultiModalModel,MultiModalModelR
from data.dataset import MultiModalDataset
from data.preprocessing import preprocess_demo, preprocess_cgm, preprocess_img
from models.multimodal_v import MultiModalModel
from models.multimodal_r import MultiModalModelR
from utils.losses import RMSRELoss
from google.colab import drive
import yaml



def train(config):
    drive.mount('/content/drive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cgm_data = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/cgm_train.csv")
    demo_data = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/demo_viome_train.csv")
    img_data = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/img_train.csv")
    label_data = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/label_train.csv")

    demo_data = preprocess_demo(demo_data)
    cgm_data = preprocess_cgm(cgm_data)
    img_data = preprocess_img(img_data)


    daily_data = pd.merge(cgm_data, img_data, on=["Subject ID", "Day"], how="inner")
    daily_data = pd.merge(daily_data, label_data, on=["Subject ID", "Day"], how="inner")
    final_data = pd.merge(demo_data, daily_data, on="Subject ID", how="inner")
    
    cgm_features = final_data['CGM Data']
    
    viome_columns = ['Viome_0', 'Viome_1', 'Viome_2', 'Viome_3', 'Viome_4', 'Viome_5',
                     'Viome_6', 'Viome_7', 'Viome_8', 'Viome_9', 'Viome_10', 'Viome_11',
                     'Viome_12', 'Viome_13', 'Viome_14', 'Viome_15', 'Viome_16', 'Viome_17',
                     'Viome_18', 'Viome_19', 'Viome_20', 'Viome_21', 'Viome_22', 'Viome_23',
                     'Viome_24', 'Viome_25', 'Viome_26']
    viome_features = final_data[viome_columns]
    pca = PCA(n_components=config['data']['pca_components'])
    viome_pca = pca.fit_transform(viome_features)
    viome_pca_df = pd.DataFrame(viome_pca, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    textual_features = pd.concat([final_data[['Age', 'Weight', 'Height', 'A1C',
           'Baseline Fasting Glucose', 'Insulin', 'Triglycerides', 'Cholesterol',
           'HDL', 'Non-HDL', 'LDL', 'VLDL', 'CHO/HDL Ratio', 'HOMA-IR', 'BMI',
           'Gender_1', 'Race_Hispanic/Latino',
           'Race_White', 'Diabetes Status_2', 'Diabetes Status_3',
           'Breakfast Time', 'Lunch Time','Breakfast Fiber',
           'Lunch Fiber']], viome_pca_df], axis=1)
    
    img_features = final_data[['Image Before Breakfast', 'Image Before Lunch']]
    
    labels = final_data[['Breakfast Calories','Lunch Calories']]

    subject_ids = final_data['Subject ID'].unique()
    
    validation_subject_ids = np.random.choice(subject_ids, 4, replace=False)
    
    train_mask = ~final_data['Subject ID'].isin(validation_subject_ids)
    val_mask = final_data['Subject ID'].isin(validation_subject_ids)
    
    train_indices = final_data.index[train_mask].tolist()
    val_indices = final_data.index[val_mask].tolist()

    train_dataset = MultiModalDataset(cgm=cgm_features.iloc[train_indices].tolist(),
                                   textual=textual_features.iloc[train_indices],
                                   img=img_features.iloc[train_indices],
                                   labels=labels.iloc[train_indices]['Lunch Calories'])
    val_dataset = MultiModalDataset(cgm=cgm_features.iloc[val_indices].tolist(),
                                       textual=textual_features.iloc[val_indices],
                                       img=img_features.iloc[val_indices],
                                       labels=labels.iloc[val_indices]['Lunch Calories'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    # model = MultiModalModelR(cgm_seq_len=config['data']['sequence_length'], textual_features=textual_features.shape[1], cnn_pretrained=True, cnn_num_blocks=None, cnn_freeze_backbone=True, cnn_output_dim=32 ).to(device)
    model = MultiModalModel(cgm_seq_len=config['data']['sequence_length'], textual_features=textual_features.shape[1], 
                          vit_pretrained=config['model']['vit_pretrained'], vit_output_dim=config['model']['vit_output_dim'], 
                          vit_freeze_backbone=config['model']['vit_freeze_backbone']).to(device)
    rmsre_criterion = RMSRELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    train_rmsre_history = []
    val_rmsre_history = []
    
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        running_rmsre = 0.0
        for cgm_seq, textual, img_bf, img_ln, labels in train_loader:
            cgm_seq, textual, img_bf, img_ln, labels = cgm_seq.to(device), textual.to(device), img_bf.to(device), img_ln.to(device), labels.to(device)
    
            outputs = model(cgm_seq, textual, img_bf, img_ln).squeeze()
            rmsre_loss = rmsre_criterion(outputs, labels)
            optimizer.zero_grad()
            rmsre_loss.backward()
            optimizer.step()
    
            running_rmsre += rmsre_loss.item()
    
        avg_train_rmsre = running_rmsre / len(train_loader)
        train_rmsre_history.append(avg_train_rmsre)
        print(f"Epoch {epoch+1}/{epochs}\nTraining RMSRE: {avg_train_rmsre:.4f}")
    
        model.eval()
        val_rmsre = 0.0
        with torch.no_grad():
            for cgm_seq, textual, img_bf, img_ln, labels in val_loader:
                cgm_seq, textual, img_bf, img_ln, labels = cgm_seq.to(device), textual.to(device), img_bf.to(device), img_ln.to(device), labels.to(device)
                outputs = model(cgm_seq, textual, img_bf, img_ln).squeeze()
                rmsre_loss = rmsre_criterion(outputs, labels)
                val_rmsre += rmsre_loss.item()
    
        avg_val_rmsre = val_rmsre / len(val_loader)
        val_rmsre_history.append(avg_val_rmsre)
        print(f"Validation RMSRE: {avg_val_rmsre:.4f}\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_rmsre_history, label='Training RMSRE', marker='o')
    plt.plot(range(1, epochs + 1), val_rmsre_history, label='Validation RMSRE', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('RMSRE Loss')
    plt.title('Training and Validation RMSRE Loss vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(config['training']['plot_save_path'])
    plt.show()
    
    torch.save(model.state_dict(), config['model']['save_path'])


def load_config():
    with open("/content/nutrition-calorie-estimation/config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    train(config)