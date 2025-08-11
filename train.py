import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# 导入自定义模块
from model import STComplEx, STCQA
from qa_datasets import STQADataset


def parse_args():
    parser = argparse.ArgumentParser(description="STCQA训练脚本")
    parser.add_argument('--tkbc_model', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_path', type=str, default='stcqa_model.pt')
    parser.add_argument('--lm_frozen', type=int, default=1)
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device):
    """训练一个周期"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        questions = batch['questions']
        targets = batch['targets'].to(device)

        optimizer.zero_grad()
        scores = model(questions)
        loss = model.loss(scores, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, k=10):
    """评估模型"""
    model.eval()
    hits = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            questions = batch['questions']
            targets = batch['targets'].to(device)

            scores = model(questions)
            _, preds = torch.topk(scores, k=k, dim=1)

            for i in range(len(targets)):
                correct = targets[i].nonzero().squeeze()
                hits += len(set(preds[i].tolist()) & set(correct.tolist()))
                total += len(correct)

    return hits / total


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载STComplEx模型
    tkbc_model = STComplEx(sizes=(15000, 200, 600, 2500), rank=512)
    tkbc_model.load_state_dict(torch.load(args.tkbc_model))
    tkbc_model = tkbc_model.to(device)

    # 创建STCQA模型
    model = STCQA(tkbc_model, args)
    model = model.to(device)

    # 加载数据集
    train_dataset = STQADataset(args.train_data)
    valid_dataset = STQADataset(args.valid_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset._collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        collate_fn=valid_dataset._collate_fn
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    best_score = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        valid_score = evaluate(model, valid_loader, device)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Valid Hits@10: {valid_score:.4f}")

        if valid_score > best_score:
            best_score = valid_score
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model with score {valid_score:.4f}")


class STCQAPredictor:
    """STCQA预测器"""

    def __init__(self, model_path, tkbc_path):
        # 加载模型
        self.tkbc_model = STComplEx(sizes=(15000, 200, 600, 2500), rank=512)
        self.tkbc_model.load_state_dict(torch.load(tkbc_path))

        self.model = STCQA(self.tkbc_model, args=None)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # 加载实体映射
        with open('entity2id.json') as f:
            self.entity2id = json.load(f)
        self.id2entity = {v: k for k, v in self.entity2id.items()}

    def predict(self, question, top_k=5):
        """预测问题答案"""
        with torch.no_grad():
            scores = self.model([question])
            _, top_indices = torch.topk(scores, k=top_k)

        return [self.id2entity[idx.item()] for idx in top_indices[0]]

if __name__ == "__main__":
    main()