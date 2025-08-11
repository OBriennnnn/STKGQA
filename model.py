import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from stcomplex import STComplEx

class STCQA(nn.Module):
    """STCQA模型实现"""

    def __init__(self, tkbc_model, args):
        super().__init__()
        self.tkbc_model = tkbc_model
        self.args = args

        # 语言模型
        self.lm_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if args.lm_frozen:
            for param in self.lm_model.parameters():
                param.requires_grad = False

        # 维度设置
        self.tkbc_embed_dim =  self.tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embed_dim = 768

        # 投影层
        self.project_sentence = nn.Linear(self.sentence_embed_dim, self.tkbc_embed_dim)
        self.project_entity = nn.Linear(self.tkbc_embed_dim, self.tkbc_embed_dim)

        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.tkbc_embed_dim,
                nhead=args.nhead,
                dropout=args.dropout
            ),
            num_layers=args.num_transformer_layers
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.tkbc_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.tkbc_model.sizes[0])  # 实体数量
        )

        # 约束解析器
        self.constraint_parser = nn.Sequential(
            nn.Linear(self.tkbc_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 预测约束类型
        )

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, questions):
        """前向传播"""
        batch_scores = []
        for question in questions:
            # 1. 预处理
            inputs = self.tokenizer(question, return_tensors='pt')
            lm_output = self.lm_model(**inputs)
            last_hidden_state = lm_output.last_hidden_state

            # 2. 实体识别和替换 (简化实现)
            tokens = self.tokenizer.tokenize(question)
            entity_positions = [i for i, token in enumerate(tokens) if token.startswith('<')]

            # 3. 投影和替换
            projected_question = self.project_sentence(last_hidden_state)
            for pos in entity_positions:
                entity_id = hash(tokens[pos]) % self.tkbc_model.sizes[0]
                entity_embed = self.tkbc_model.embeddings[0](torch.tensor([entity_id]))
                projected_question[0, pos] = self.project_entity(entity_embed)

            # 4. Transformer融合
            transformer_output = self.transformer_encoder(projected_question)
            cls_embedding = transformer_output[0, 0, :]

            # 5. 答案预测
            scores = self.classifier(cls_embedding)
            batch_scores.append(scores)

        return torch.stack(batch_scores)

    def loss(self, scores, targets):
        """计算损失"""
        return self.loss_fn(scores, targets)