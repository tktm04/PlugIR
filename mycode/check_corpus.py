import torch

# 保存されたファイルをロードする
corpus_vectors_path = 'corpus_vectors_clip.pt'

corpus_vectors = torch.load(corpus_vectors_path)

# 特徴ベクトルのサイズを確認
print(f"Loaded corpus_vectors shape: {corpus_vectors.shape}")

# 50000個の特徴ベクトルが存在するか確認
if corpus_vectors.shape[0] == 50000:
    print("50000個の特徴ベクトルが正常に保存されています。")
else:
    print(f"特徴ベクトルの数: {corpus_vectors.shape[0]}。50000個ではありません。")
 

