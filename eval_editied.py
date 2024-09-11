import torch
import tqdm
import os.path
import json
import csv
from PIL import Image
import torch.nn.functional as F
from transformers import AutoProcessor, BlipForImageTextRetrieval
import argparse
import clip
from torch.nn.functional import normalize
from typing import Any, Optional
import logging
from finetune import utils

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--retriever', type=str, default='clip', choices=["clip", "blip"])
parser.add_argument('--queries-path', type=str, default='dialogues/VisDial_v1.0_queries_val.json')
parser.add_argument('--ft-model-path', type=str)
parser.add_argument('--cache-corpus', type=str)
parser.add_argument('--data-dir', type=str, default='visdial/corpus')
parser.add_argument('--output-dir', type=str, default='logs')
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--num-rounds', type=int, default=11)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--split', action='store_true', help="load dialog (caption) in split")

cfg = {'corpus_bs': 256,
       'queries_bs': 256,
       'num_workers': 8,
       'sep_token': ', ',
       'queries_path': None,
       'corpus_path': 'Protocol/Search_Space_val_50k.json',
       'device': 'cuda:0',
       }

args = parser.parse_args()
retriever = args.retriever
queries_path = args.queries_path
cfg['data_dir'] = args.data_dir
cfg['queries_path'] = queries_path
cfg['split'] = args.split
cfg['finetuned_model_path'] = args.ft_model_path
cfg['cache_corpus'] = args.cache_corpus
cfg['K'] = args.K
cfg['queries_bs'] = args.batch_size
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.output_dir:
    utils.mkdir(args.output_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.output_dir, 'test.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

# 結果保存ファイル
summary_file = os.path.join(args.output_dir, f'dialog_length_results_{os.path.basename(cfg["cache_corpus"])}.csv')
ranking_changes_file = os.path.join(args.output_dir, f'target_image_ranking_changes_{os.path.basename(cfg["cache_corpus"])}.csv')


class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        return text_feat

    def get_image_features(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_feat = normalize(self.vision_proj(vision_outputs[0][:, 0, :]), dim=-1)
        return image_feat


class ImageEmbedder:
    def __init__(self, model, preprocessor):
        self.model = model
        self.processor = preprocessor


class Corpus(torch.utils.data.Dataset):
    def __init__(self, data_dir, corpus_path, preprocessor):
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        self.corpus = [os.path.join(data_dir, path) for path in self.corpus]
        self.preprocessor = preprocessor
        self.path2id = {self.corpus[i]: i for i in range(len(self.corpus))}

    def __len__(self):
        return len(self.corpus)

    def path_to_index(self, path):
        return self.path2id[path]

    def __getitem__(self, i):
        if retriever == 'blip':
            image = self.preprocessor(self.corpus[i])['pixel_values'][0]
        else:
            image = self.preprocessor(self.corpus[i])
        return {'id': i, 'image': image}


class Queries(torch.utils.data.Dataset):
    def __init__(self, cfg, queries_path, txt_processors):
        with open(queries_path) as f:
            self.queries = json.load(f)
        self.dialog_length = None
        self.cfg = cfg
        self.txt_processors = txt_processors

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        target_path = os.path.join(self.cfg['data_dir'], self.queries[i]['img'])
        if self.cfg['split']:
            text = self.queries[i]['dialog'][self.dialog_length]
        else:
            text = self.cfg['sep_token'].join(self.queries[i]['dialog'][:self.dialog_length + 1])
        return {'text': text, 'target_path': target_path}


class PlugIREval:
    def __init__(self, cfg, dialog_encoder, image_embedder: ImageEmbedder, txt_processors):
        self.dialog_encoder = dialog_encoder
        self.image_embedder = image_embedder
        self.txt_processors = txt_processors

        self.cfg = cfg
        self.corpus = None
        self.corpus_dataset = Corpus(self.cfg['data_dir'], self.cfg['corpus_path'], self.image_embedder.processor)
        self.scores = {}
        self.ranks = []
        self.targets = []

    def _get_recalls(self, dataloader, dialog_length):
        dataloader.dataset.dialog_length = dialog_length
        recalls = []
        ranks = []
        targets = []

        for i, batch in enumerate(tqdm.tqdm(dataloader)):
            target_ids = torch.tensor([self.corpus_dataset.path_to_index(p) for p in batch['target_path']]).unsqueeze(1).to(self.cfg['device'])
            pred_vec = F.normalize(self.dialog_encoder(batch['text']), dim=-1)  # Nxd
            self.scores[i] = pred_vec @ self.corpus[1].T
            arg_ranks = torch.argsort(self.scores[i], descending=True, dim=1).long()
            target_recall = ((arg_ranks - target_ids) == 0).nonzero()[:, 1]
            recalls.append(target_recall)
            ranks.append(arg_ranks)
            targets.append(target_ids)
            self.scores[i] = self.scores[i].cpu()

        return torch.cat(recalls).cpu(), torch.cat(ranks).cpu(), torch.cat(targets).cpu()

    def run(self, hits_at):
        assert self.corpus, f"Prepare corpus first (self.index_corpus())"
        dataset = Queries(cfg, self.cfg['queries_path'], self.txt_processors)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.cfg['queries_bs'],
                                                 shuffle=False,
                                                 num_workers=self.cfg['num_workers'],
                                                 pin_memory=True,
                                                 drop_last=False
                                                 )
        hits_results = []
        ranks_results = []
        targets_results = []
        min_ranks = []
        rank_changes = {}  # ターゲット画像ごとの順位変動を追跡

        for dl in range(args.num_rounds):
            logging.info(f"Calculate recalls for each dialogues of length {dl}...")
            dialog_recalls, ranks, targets = self._get_recalls(dataloader, dialog_length=dl)

            # ターゲット画像の順位変動を記録
            for i, target_id in enumerate(targets):
                target_image_id = self.corpus_dataset.corpus[target_id.item()]
                if target_image_id not in rank_changes:
                    rank_changes[target_image_id] = [None] * args.num_rounds  # ラウンドごとの順位を保存
                rank_changes[target_image_id][dl] = ranks[i].tolist().index(target_id.item())  # ラウンドでの順位

            if dl == 0:
                min_ranks.append(dialog_recalls)
            else:
                min_ranks.append(torch.minimum(min_ranks[dl-1], dialog_recalls))

            hits_results.append(dialog_recalls)
            ranks_results.append(ranks)
            targets_results.append(targets)

        hits_results, temp_hits_results = cumulative_hits_per_round(
            torch.cat(hits_results),
            torch.cat(ranks_results),
            torch.cat(targets_results),
            hitting_recall=cfg['K'])

        # CSVファイルに順位変動を保存
        with open(os.path.join(args.output_dir, 'target_image_ranking_changes.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Target Image ID'] + [f'Rank at Round {i}' for i in range(args.num_rounds)])
            for target_image_id, ranks in rank_changes.items():
                writer.writerow([target_image_id] + ranks)

        # その他のログ出力はそのまま
        hits_results = hits_results.tolist()
        temp_hits_results = temp_hits_results.tolist()
        logging.info(f"====== Results for Hits@{cfg['K']} ====== ")
        for dl in range(args.num_rounds):
            logging.info(f"\t Dialog Length: {dl}: {round(hits_results[dl], 2)}%")
        logging.info(f"====== Results for Recall@{cfg['K']} ====== ")
        for dl in range(args.num_rounds):
            logging.info(f"\t Dialog Length: {dl}: {round(temp_hits_results[dl], 2)}%")
        logging.info(f"====== Best log Rank Integral ====== ")
        bri = 0
        for dl in range(args.num_rounds-1):
            bri += ((min_ranks[dl] + min_ranks[dl+1]) / 2).mean()
        bri /= args.num_rounds-1
        logging.info(f"\t BRI: {bri}")


def cumulative_hits_per_round(target_recall, ranks, targets, hitting_recall=10):
    ht_times, temp_ht_times = get_first_hitting_time(target_recall, hitting_recall)
    return ((ht_times < torch.inf).sum(dim=-1) * 100 / ht_times[0].shape[0]), ((temp_ht_times < torch.inf).sum(dim=-1) * 100 / temp_ht_times[0].shape[0])


def get_first_hitting_time(target_recall, hitting_recall=10):
    target_recalls = target_recall.view(args.num_rounds, -1).T
    hits = (target_recalls < hitting_recall)

    final_hits = torch.inf * torch.ones(target_recalls.shape[0])
    hitting_times = []
    temp_hitting_times = []
    for ro_i in range(args.num_rounds):
        temp_hits = torch.inf * torch.ones(target_recalls.shape[0])
        rh = hits[:, ro_i]
        final_hits[rh] = torch.min(final_hits[rh], torch.ones(final_hits[rh].shape) * ro_i)
        temp_hits[rh] = torch.min(temp_hits[rh], torch.ones(temp_hits[rh].shape) * ro_i)
        hitting_times.append(final_hits.clone())
        temp_hitting_times.append(temp_hits)

    return torch.stack(hitting_times), torch.stack(temp_hitting_times)


with torch.no_grad():
    txt_processors = None
    if retriever == 'blip':
        dialog_encoder, image_embedder = BLIP_ZERO_SHOT_BASELINE(cfg)
    else:
        dialog_encoder, image_embedder = CLIP_ZERO_SHOT_BASELINE()
    evaluator = PlugIREval(cfg, dialog_encoder, image_embedder, txt_processors)
    evaluator.index_corpus()
    evaluator.run(hits_at=cfg['K'])
