import json
import torch
from torch.utils.data import Dataset


class STQADataset(Dataset):
    """STQA数据集加载器"""

    def __init__(self, file_path):
        with open(file_path) as f:
            self.data = json.load(f)

        # 构建实体到ID的映射
        self.entity2id = {e: i for i, e in enumerate(self._collect_entities())}
        self.num_entities = len(self.entity2id)

    def _collect_entities(self):
        """收集所有唯一实体"""
        entities = set()
        for item in self.data:
            entities.update(item['entities'])
            entities.update(item['answers'])
        return list(entities)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'question': item['question'],
            'entities': item['entities'],
            'answers': [self.entity2id[a] for a in item['answers']],
            'type': item['type']
        }

    def _collate_fn(self, batch):
        """批处理函数"""
        questions = [item['question'] for item in batch]
        answers = [item['answers'] for item in batch]

        # 创建多热编码目标
        targets = torch.zeros(len(batch), self.num_entities)
        for i, ans_ids in enumerate(answers):
            targets[i, ans_ids] = 1.0

        return {
            'questions': questions,
            'targets': targets
        }