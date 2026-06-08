"""
CS2 round prediction dataset
Supports single-frame and sequence modes, Direction 3-class and Winner 2-class.
"""
import json
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


DIRECTION_MAP = {"A": 0, "B": 1, "none": 2}
WINNER_MAP = {"ct": 0, "t": 1}


def default_transform(img_size: int, is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class CS2RoundDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        plant_times_path: str,
        split: str = "train",
        task: str = "direction",
        max_frames: int = 20,
        mode: str = "single",
        img_size: int = 224,
        transform=None,
        split_ratio: float = 0.7,
        seed: int = 42,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.task = task
        self.max_frames = max_frames
        self.mode = mode
        self.img_size = img_size
        
        if transform is None:
            self.transform = default_transform(img_size, is_train=(split == "train"))
        else:
            self.transform = transform
        
        with open(plant_times_path) as f:
            self.plant_times = json.load(f)
        
        self.rounds = self._collect_rounds()
        self.rounds = self._split_by_match(self.rounds, split_ratio, seed)
        
        print(f"[{split}] {task} | {mode} | Loaded {len(self.rounds)} rounds")
    
    def _collect_rounds(self):
        rounds = []
        
        for match_dir in sorted(self.dataset_dir.iterdir()):
            if not match_dir.is_dir():
                continue
            match_name = match_dir.name
            
            for map_dir in sorted(match_dir.iterdir()):
                if not map_dir.is_dir():
                    continue
                map_name = map_dir.name
                
                metadata_path = map_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                demo_name = metadata["dataset_info"]["demo_name"]
                
                round_samples = {}
                for sample in metadata.get("samples", []):
                    rnum = sample["round_number"]
                    if rnum not in round_samples:
                        round_samples[rnum] = []
                    
                    img_rel = sample["image_path"]
                    img_path = self.dataset_dir / match_name / map_name / img_rel
                    
                    round_samples[rnum].append({
                        "img_path": str(img_path),
                        "rel_time": float(sample["round_relative_time"]),
                        "direction": sample["direction"],
                        "winner": sample["winner"],
                        "alive_t": sample["num_players_alive_t"],
                        "alive_ct": sample["num_players_alive_ct"],
                    })
                
                for rnum, frames in round_samples.items():
                    frames.sort(key=lambda x: x["rel_time"])
                    
                    plant_info = self.plant_times.get(demo_name, {}).get(str(rnum), {})
                    plant_time = plant_info.get("plant_relative_time")
                    direction = plant_info.get("direction", "none")
                    winner = plant_info.get("winner", "")
                    
                    if plant_time is not None:
                        valid_frames = [f for f in frames if f["rel_time"] < plant_time - 10]
                    else:
                        valid_frames = frames
                    
                    if len(valid_frames) == 0:
                        continue
                    
                    selected = valid_frames[-self.max_frames:]
                    
                    rounds.append({
                        "match_name": match_name,
                        "map_name": map_name,
                        "demo_name": demo_name,
                        "round_number": rnum,
                        "frames": selected,
                        "direction": direction,
                        "winner": winner,
                        "has_plant": plant_time is not None,
                        "plant_time": plant_time,
                    })
        
        return rounds
    
    def _split_by_match(self, rounds, ratio, seed):
        match_rounds = {}
        for r in rounds:
            m = r["match_name"]
            if m not in match_rounds:
                match_rounds[m] = []
            match_rounds[m].append(r)
        
        random.seed(seed)
        match_names = sorted(match_rounds.keys())
        random.shuffle(match_names)
        
        n_train = int(len(match_names) * ratio)
        train_matches = set(match_names[:n_train])
        
        if self.split == "train":
            result = []
            for m in train_matches:
                result.extend(match_rounds[m])
            return result
        else:
            result = []
            for m in match_names[n_train:]:
                result.extend(match_rounds[m])
            return result
    
    def __len__(self):
        return len(self.rounds)
    
    def __getitem__(self, idx):
        round_info = self.rounds[idx]
        frames = round_info["frames"]
        
        imgs = []
        for f in frames:
            img = Image.open(f["img_path"]).convert("RGB")
            img = self.transform(img)
            imgs.append(img)
        
        if self.mode == "single":
            idx = random.randint(0, len(imgs) - 1)
            img = imgs[idx]
            
            if self.task == "direction":
                label = DIRECTION_MAP[round_info["direction"]]
                return img, label, round_info["has_plant"]
            elif self.task == "winner":
                label = WINNER_MAP.get(round_info["winner"].lower(), -1)
                return img, label, round_info["has_plant"]
            else:
                raise ValueError(f"Unknown task: {self.task}")
        else:
            seq = torch.stack(imgs)
            
            if self.task == "direction":
                label = DIRECTION_MAP[round_info["direction"]]
                return seq, label, round_info["has_plant"]
            elif self.task == "winner":
                label = WINNER_MAP.get(round_info["winner"].lower(), -1)
                return seq, label, round_info["has_plant"]
            else:
                raise ValueError(f"Unknown task: {self.task}")
