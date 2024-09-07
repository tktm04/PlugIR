import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
from tqdm import tqdm  # 進捗表示のためにtqdmをインポート
import gc  # ガベージコレクション

# コーパス画像のロード
def load_corpus_images(corpus_file_path, data_dir):
    with open(corpus_file_path, 'r') as f:
        corpus_images = json.load(f)
    corpus_image_paths = [os.path.join(data_dir, img) for img in corpus_images]
    return corpus_image_paths

# キャプションのロード
def load_captions(captions_file_path):
    with open(captions_file_path, 'r') as f:
        captions_data = json.load(f)
    image_to_caption = {entry['id']: entry['caption'] for entry in captions_data}
    return image_to_caption

# Datasetクラスの定義
class CorpusDatasetBLIP(Dataset):
    def __init__(self, image_paths, image_to_caption, processor):
        self.image_paths = image_paths
        self.image_to_caption = image_to_caption
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image_id = os.path.basename(image_path)
        caption = self.image_to_caption.get(image_id, ["No caption available"])
        processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        return {'image': processed_image, 'caption': caption}

# DataLoaderの作成
def create_dataloader(dataset, batch_size=1, num_workers=0):  # メモリ削減のためにバッチサイズとワーカー数を調整
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

# コーパスベクトルの保存と中間ファイルの作成
def save_corpus_vectors(dataloader, model, device, output_path='corpus_vectors_blip.pt', checkpoint_file='checkpoint.json'):
    corpus_vectors = []
    last_saved_batch = 0

    # チェックポイントの読み込み
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            last_saved_batch = checkpoint.get("last_saved_batch", 0)
            # ファイルが存在する場合のみ読み込み
            if os.path.exists(checkpoint.get("corpus_file", output_path)):
                corpus_vectors = torch.load(checkpoint.get("corpus_file", output_path))
            else:
                print(f"Warning: {checkpoint.get('corpus_file', output_path)} not found, starting from scratch.")
    
    # バッチ処理を再開
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches", initial=last_saved_batch)):
        if i < last_saved_batch:
            continue  # 再開時に処理済みのバッチをスキップ

        images = batch['image'].to(device)
        
        # BLIPのモデルを使って特徴量を抽出
        outputs = model.vision_model(pixel_values=images)
        image_embeds = outputs.last_hidden_state[:, 0, :]  # [CLS]トークンの出力を特徴量として使用
        corpus_vectors.append(image_embeds)
        
        # メモリを解放
        del images, outputs, image_embeds
        torch.cuda.empty_cache()  # GPUキャッシュの解放
        gc.collect()  # ガベージコレクションの強制実行

        # 50バッチごとに保存
        if (i + 1) % 50 == 0:
            corpus_vectors_tensor = torch.cat(corpus_vectors)
            torch.save(corpus_vectors_tensor, output_path)
            with open(checkpoint_file, 'w') as f:
                json.dump({"last_saved_batch": i + 1, "corpus_file": output_path}, f)
            corpus_vectors = []  # メモリ使用量を抑えるためリセット
            torch.cuda.empty_cache()  # キャッシュを再度解放

    # 最終的に全てのベクトルを保存
    if corpus_vectors:
        corpus_vectors_tensor = torch.cat(corpus_vectors)
        torch.save(corpus_vectors_tensor, output_path)
        with open(checkpoint_file, 'w') as f:
            json.dump({"last_saved_batch": i + 1, "corpus_file": output_path}, f)

# BLIPを使ったコーパス生成
def generate_blip_corpus(corpus_file_path, captions_file_path, data_dir):
    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to('cuda' if torch.cuda.is_available() else 'cpu')

    # コーパスとキャプションをロードしてデータセットを作成
    corpus_image_paths = load_corpus_images(corpus_file_path, data_dir)
    image_to_caption = load_captions(captions_file_path)
    dataset = CorpusDatasetBLIP(corpus_image_paths, image_to_caption, processor)

    # DataLoaderの作成
    dataloader = create_dataloader(dataset)

    # コーパスベクトルの保存 (進捗が表示される)
    save_corpus_vectors(dataloader, model, 'cuda' if torch.cuda.is_available() else 'cpu')

# 実行例
if __name__ == "__main__":
    corpus_file_path = '../Protocol/Search_Space_val_50k.json'
    captions_file_path = '../Protocol/visdial_captions.json'
    data_dir = '../../Visdial/'
    
    generate_blip_corpus(corpus_file_path, captions_file_path, data_dir)
