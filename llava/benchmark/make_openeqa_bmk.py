import os
import json
from tqdm import tqdm
import shutil
import pandas as pd

class ParquetDataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        initial_data = {
            "id": [],
            "type": [],
            "question": [],
            "images": [],
            "answer": [],
            "extra_answers": []
        }
        self.df = pd.DataFrame(initial_data)

    def load_from_parquet(self):
        self.df = pd.read_parquet(self.file_path, engine='pyarrow')

    def add_data(self, new_data):
        new_df = pd.DataFrame(new_data)
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def export_to_parquet(self):
        self.df.to_parquet(self.file_path, engine='pyarrow')

    def get_length(self):
      return len(self.df)

#@title Task utils
"""Tasks related utils."""

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def extract_images(input_dir, output_dir):
    for root, dirs, files in tqdm(os.walk(input_dir)):
        rgb_files = sorted([f for f in files if f.endswith('-rgb.png')])
        
        if rgb_files:
            step = max(1, len(rgb_files) // 32)
            selected_files = rgb_files[::step][:32]
            
            output_path = os.path.join(output_dir, os.path.relpath(root, input_dir))
            os.makedirs(output_path, exist_ok=True)
            
            for file_name in selected_files:
                src_file = os.path.join(root, file_name)
                dst_file = os.path.join(output_path, file_name)
                shutil.copy2(src_file, dst_file)
                # print(f"Copied {src_file} to {dst_file}")

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
  
def make_bench():
  bench_json = "./open-eqa-v0.json"
  scannet_parquet_path = './OpenEQA/openeqa_scannet.parquet'
  hm3d_parquet_path = './OpenEQA/openeqa_hm3d.parquet'
  scannet_handler = ParquetDataHandler(scannet_parquet_path)
  hm3d_handler = ParquetDataHandler(hm3d_parquet_path)
  
  json_data = read_json(bench_json)

  for item in tqdm(json_data):
    images = []
    if item["episode_history"].split('/')[0] == 'hm3d-v0':
      item_path = os.path.join('./openeqa_val', item["episode_history"])
      item_path_rec = os.path.join('openeqa_val', item["episode_history"])
      imgs_list = os.listdir(item_path)
      
      for name in imgs_list:
        images.append(os.path.join(item_path_rec, name))
        
      new_data = {   
            "id": [item['question_id']],
            "type": [f"openeqa_hm3d-v0_{item['category'].replace(' ','-')}"],
            "question": [item['question']],
            "images": [images],
            "answer": [item['answer']],
            "extra_answers": [item.get('extra_answers', None)]
        }
      hm3d_handler.add_data(new_data)
      
    elif item["episode_history"].split('/')[0] == 'scannet-v0':
      item_path = os.path.join('./openeqa_val', item["episode_history"])
      item_path_rec = os.path.join('openeqa_val', item["episode_history"])
      imgs_list = os.listdir(item_path)
      
      for name in imgs_list:
        images.append(os.path.join(item_path_rec, name))
        
      new_data = {   
            "id": [item['question_id']],
            "type": [f"openeqa_scannet-v0_{item['category'].replace(' ','-')}"],
            "question": [item['question']],
            "images": [images],
            "answer": [item['answer']],
            "extra_answers": [item.get('extra_answers', None)]
        }
      scannet_handler.add_data(new_data)
      
    else:
      raise NameError('Invaild Dataset Name !!!')
    
  print(f"hm3d: {hm3d_handler.get_length()}")
  print(f"scannet: {scannet_handler.get_length()}")
  
  os.makedirs('./OpenEQA', exist_ok=True)
  print("====== Exporting openeqa_hm3d.parquet ...")
  hm3d_handler.export_to_parquet()
  print("====== Exporting openeqa_scannet.parquet ...")
  scannet_handler.export_to_parquet()

if __name__ == '__main__':
  
  print("====== Extracting Images ...")
  extract_images('./frames', './openeqa_val')
  
  print("====== Making Benchmark ...")
  make_bench()
  
  print("====== Evaluate Scannet Benchmark ...")
  handler = ParquetDataHandler('./OpenEQA/openeqa_scannet.parquet')
  handler.load_from_parquet()
  row = handler.df.iloc[0]
  print(row)
  row = handler.df.iloc[-1]
  print(row)
  print(handler.get_length())
  
  print("====== Evaluate HM3D Benchmark ...")
  handler = ParquetDataHandler('./OpenEQA/openeqa_hm3d.parquet')
  handler.load_from_parquet()
  row = handler.df.iloc[0]
  print(row)
  row = handler.df.iloc[-1]
  print(row)
  print(handler.get_length())