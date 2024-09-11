import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import clip
from tqdm import tqdm
import torch.nn.functional as F

# コーパス画像のロード
def load_corpus_images(corpus_file_path, data_dir):
    with open(corpus_file_path, 'r') as f:
        corpus_images = json.load(f)
    corpus_image_paths = [os.path.join(data_dir, img) for img in corpus_images]
    return corpus_image_paths

# Datasetクラスの定義
class CorpusDatasetCLIP(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")  # 画像をRGBに変換
            processed_image = self.preprocess(image)  # CLIPのpreprocessを使用
            return {'image': processed_image, 'id': idx}
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

# DataLoaderの作成
def create_dataloader(dataset, batch_size=32, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# コーパスベクトルの作成と保存
def save_corpus_vectors(dataloader, model, device, output_path='corpus_vectors_clip.pt'):
    corpus_vectors = []
    corpus_ids = []

    for batch in tqdm(dataloader, desc="Processing batches"):
        if batch is None:
            continue
        
        images = batch['image'].to(device)
        ids = batch['id'].to(device)

        # CLIPのモデルを使って特徴量を抽出し、正規化
        with torch.no_grad():
            image_embeds = model.encode_image(images).to(torch.float32)  # float32に変換
            image_embeds = F.normalize(image_embeds, dim=-1)  # 正規化を追加

        # ベクトルとIDを追加
        corpus_vectors.append(image_embeds)
        corpus_ids.append(ids)

    # 全てのベクトルとIDを連結
    corpus_vectors = torch.cat(corpus_vectors)
    corpus_ids = torch.cat(corpus_ids)

    # IDでソート
    arg_ids = torch.argsort(corpus_ids)
    corpus_vectors = corpus_vectors[arg_ids]  # GPU上にそのまま保持
    corpus_ids = corpus_ids[arg_ids]

    # コーパスを保存
    torch.save((corpus_ids, corpus_vectors), output_path)
    print(f"Saved corpus vectors to {output_path}")


# CLIPを使ったコーパス生成
def generate_clip_corpus(corpus_file_path, data_dir, cache_dir):
    # CLIPモデルとプロセッサのロード
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)

    # コーパスをロードしてデータセットを作成
    corpus_image_paths = load_corpus_images(corpus_file_path, data_dir)
    dataset = CorpusDatasetCLIP(corpus_image_paths, preprocess)

    # DataLoaderの作成
    dataloader = create_dataloader(dataset)
    
    output_path = os.path.join(cache_dir, 'corpus_vectors_clip.pt')

    # コーパスベクトルの保存
    save_corpus_vectors(dataloader, model, device, output_path)

# 実行例
if __name__ == "__main__":
    corpus_file_path = '../Protocol/Search_Space_val_50k.json'
    data_dir = '../../Visdial/'
    cache_dir = '../cache/'
    
    generate_clip_corpus(corpus_file_path, data_dir, cahce_dir)
