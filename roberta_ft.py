import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from tqdm import tqdm
import argparse


class Sentiment140Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['sentiment']

        label = 0 if label == 0 else 1

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


class SentimentClassifier(nn.Module):
    def __init__(self, roberta_model):
        super(SentimentClassifier, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(roberta_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # binary for now
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def save_classifier_head(model, save_path):
    torch.save(model.module.classifier.state_dict(), save_path)

def main(rank, world_size, batch_size, num_epochs):
    setup(rank, world_size)

    dataset = load_dataset('sentiment140')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    train_dataset = Sentiment140Dataset(dataset['train'], tokenizer)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)

    pretrained_model_name = 'roberta-large'
    roberta_model = RobertaModel.from_pretrained(pretrained_model_name)

    freeze_model_parameters(roberta_model)

    model = SentimentClassifier(roberta_model).to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = AdamW(model.module.classifier.parameters(), lr=1e-3)

    # train loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=rank)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(rank)
            attention_mask = batch['attention_mask'].to(rank)
            labels = batch['labels'].to(rank)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=avg_loss)

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

    if rank == 0:
        save_classifier_head(model, 'classifier_head.pt')

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=4, help='number of gpus')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size per gpu')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    args = parser.parse_args()

    world_size = args.world_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    torch.multiprocessing.spawn(main, args=(world_size, batch_size, num_epochs), nprocs=world_size, join=True)
    

