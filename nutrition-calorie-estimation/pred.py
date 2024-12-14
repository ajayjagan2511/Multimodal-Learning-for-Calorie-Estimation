import torch
import yaml
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from models import MultiModalModel
from models import MultiModalModelR
from data.dataset import MultiModalDataset
from torch.utils.data import DataLoader
from data.preprocessing import preprocess_demo, preprocess_cgm, preprocess_img
from utils.image_utils import transform_images
from utils.time_utils import time_to_seconds
from utils.validation_utils import is_invalid_before_breakfast
from google.colab import drive
import argparse

def predict(config):
    drive.mount('/content/drive')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgm_test = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/cgm_test.csv")
    demo_viome_test = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/demo_viome_test.csv")
    img_test = pd.read_csv("/content/drive/MyDrive/fa-24-tamu-csce-633-600-machine-learning/img_test.csv")

    demo_viome_test = preprocess_demo(demo_viome_test)
    cgm_test = preprocess_cgm(cgm_test)
    img_test = preprocess_img(img_test)
    final_test_data = pd.merge(demo_viome_test, cgm_test, on="Subject ID", how="inner")
    final_test_data = pd.merge(final_test_data, img_test, on=["Subject ID", "Day"], how="inner")

    test_cgm_features = final_test_data['CGM Data'].tolist()

    test_viome_columns = ['Viome_0', 'Viome_1', 'Viome_2', 'Viome_3', 'Viome_4', 'Viome_5',
                     'Viome_6', 'Viome_7', 'Viome_8', 'Viome_9', 'Viome_10', 'Viome_11',
                     'Viome_12', 'Viome_13', 'Viome_14', 'Viome_15', 'Viome_16', 'Viome_17',
                     'Viome_18', 'Viome_19', 'Viome_20', 'Viome_21', 'Viome_22', 'Viome_23',
                     'Viome_24', 'Viome_25', 'Viome_26']
    test_viome_features = final_test_data[test_viome_columns]
    test_pca = PCA(n_components=config['data']['pca_components'])    
    test_viome_pca = test_pca.fit_transform(test_viome_features)
    test_viome_pca_df = pd.DataFrame(test_viome_pca, columns=[f'PC{i+1}' for i in range(test_pca.n_components_)])
    test_textual_features = pd.concat([final_test_data[['Age', 'Weight', 'Height', 'A1C',
           'Baseline Fasting Glucose', 'Insulin', 'Triglycerides', 'Cholesterol',
           'HDL', 'Non-HDL', 'LDL', 'VLDL', 'CHO/HDL Ratio', 'HOMA-IR', 'BMI',
           'Gender_1', 'Race_Hispanic/Latino',
           'Race_White', 'Diabetes Status_2', 'Diabetes Status_3',
           'Breakfast Time', 'Lunch Time','Breakfast Fiber',
           'Lunch Fiber']], test_viome_pca_df], axis=1)
    
    test_img_features = final_test_data[['Image Before Breakfast', 'Image Before Lunch']]

    test_dataset = MultiModalDataset(cgm=test_cgm_features, textual=test_textual_features,img=test_img_features, labels=None)
    test_loader = DataLoader(test_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])

    model = MultiModalModel(cgm_seq_len=config['data']['sequence_length'], textual_features=test_textual_features.shape[1], 
                          vit_pretrained=config['model']['vit_pretrained'], vit_output_dim=config['model']['vit_output_dim'], 
                          vit_freeze_backbone=config['model']['vit_freeze_backbone']).to(device)
    
    model.load_state_dict(torch.load(config['model']['save_path'], weights_only=True))
    model.eval()
    
    
    predictions = []
    with torch.no_grad():
        for cgm_seq, textual, img_bf, img_ln in test_loader:
            cgm_seq, textual, img_bf, img_ln = cgm_seq.to(device), textual.to(device), img_bf.to(device), img_ln.to(device)
            output = model(cgm_seq, textual, img_bf, img_ln)
            predictions.append(output.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    lunch_calories = predictions
    
    result_df = pd.DataFrame(lunch_calories, columns=["label"])
    
    result_df["row_id"] = result_df.index
    
    result_df = result_df[["row_id", "label"]]
    
    result_df.to_csv(config['prediction']['save_path'], index=False)
    pass

def load_config():
    with open("/content/nutrition-calorie-estimation/config/config.yaml", "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config()
    predict(config)