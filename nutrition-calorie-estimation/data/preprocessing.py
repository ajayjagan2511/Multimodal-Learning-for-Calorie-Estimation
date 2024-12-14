import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from ast import literal_eval
import torch
from torchvision import transforms
from utils import transform_images, time_to_seconds


def preprocess_demo(demo_data):
  categorical_columns = ['Gender', 'Race', 'Diabetes Status']

  demo_data['Viome'] = demo_data['Viome'].apply(str.split, args=(',',))
  demo_data['Viome'] = demo_data['Viome'].apply(lambda x: list(map(float, x)))

  viome_features = pd.DataFrame(
    [x for x in demo_data['Viome']],
    columns=[f'Viome_{i}' for i in range(len(demo_data['Viome'][0]))],
    index=demo_data.index
  )

  demo_data = demo_data.drop(columns=['Viome'])
  demo_data = pd.concat([demo_data, viome_features], axis=1)

  numeric_columns = []
  for col in demo_data.columns:
    if col not in categorical_columns and col != 'Subject ID':
        numeric_columns.append(col)


  demo_data[categorical_columns]=demo_data[categorical_columns].fillna(demo_data[categorical_columns].mode().iloc[0])
  demo_data[numeric_columns]=demo_data[numeric_columns].fillna(demo_data[numeric_columns].mean())

  encoder = OneHotEncoder(sparse_output=False, drop='first')
  categorical_encoded = pd.DataFrame(
        encoder.fit_transform(demo_data[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns),
        index=demo_data.index
  )

  scaler = MinMaxScaler()
  numeric_normalized = pd.DataFrame(
        scaler.fit_transform(demo_data[numeric_columns]),
        columns=numeric_columns,
        index=demo_data.index
  )

  processed_demo_data = pd.concat([demo_data["Subject ID"],numeric_normalized, categorical_encoded], axis=1)

  return processed_demo_data






def preprocess_cgm(cgm_data, sequence_length=200):
  time_series = cgm_data['CGM Data']
  processed_data = []
  for entry in time_series:
    try:
        if isinstance(entry, str):
            entry = literal_eval(entry)
        sequence = [float(x[1]) for x in entry]
        if len(sequence) < sequence_length:
            sequence += [0.0] * (sequence_length - len(sequence))
        else:
            sequence = sequence[:sequence_length]
        processed_data.append(sequence)
    except (ValueError, TypeError) as e:
        processed_data.append([0.0] * sequence_length)

  cgm_data['CGM Data'] = processed_data
  cgm_data['Breakfast Time'] = pd.to_datetime(cgm_data['Breakfast Time'], errors='coerce')
  cgm_data['Lunch Time'] = pd.to_datetime(cgm_data['Lunch Time'], errors='coerce')

  mean_breakfast = cgm_data['Breakfast Time'].mean()
  mean_lunch = cgm_data['Lunch Time'].mean()

  cgm_data['Breakfast Time'] = cgm_data['Breakfast Time'].fillna(mean_breakfast)
  cgm_data['Lunch Time'] = cgm_data['Lunch Time'].fillna(mean_lunch)

  cgm_data['Breakfast Time'] = time_to_seconds(cgm_data['Breakfast Time'])
  cgm_data['Lunch Time'] = time_to_seconds(cgm_data['Lunch Time'])

  return cgm_data





def preprocess_img(img_data):
  invalid_before_breakfast = img_data[img_data['Image Before Breakfast'].isin(["[]", "", " "])]
  invalid_before_lunch = img_data[img_data['Image Before Lunch'].isin(["[]", "", " "])]

  transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  img_data['Image Before Breakfast'] = transform_images(img_data['Image Before Breakfast'], transform)
  img_data['Image Before Lunch'] = transform_images(img_data['Image Before Lunch'], transform)
  img_data['Breakfast Fiber'] = img_data['Breakfast Fiber'].fillna(img_data['Breakfast Fiber'].mean()).round().astype(int)
  img_data['Lunch Fiber'] = img_data['Lunch Fiber'].fillna(img_data['Lunch Fiber'].mean()).round().astype(int)

  return img_data