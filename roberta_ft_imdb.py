import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import RobertaModel, RobertaTokenizer
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import argparse


class IMDbDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']

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
        self.dropout = nn.Dropout(p=0.01)
        self.classifier = nn.Sequential(
                # too large (try 512)
            nn.Linear(roberta_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # binary
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
    #os.environ['MASTER_PORT'] = str(find_free_port())
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def save_classifier_head(model, save_path):
    torch.save(model.module.classifier.state_dict(), save_path)

def load_classifier_head(model, load_path):
    model.module.classifier.load_state_dict(torch.load(load_path))

# Not needed for now
#def find_free_port():
 #   import socket
  #  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
   #     s.bind(('', 0))
    #    return s.getsockname()[1]


class CustomLRScheduler:
    def __init__(self, optimizer, factor=1.5, patience=2):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.best_loss = float('inf')
        self.epochs_since_improvement = 0
        self.lr_history = [optimizer.param_groups[0]['lr']]

    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
            if self.epochs_since_improvement >= self.patience:
                self.adjust_learning_rate()
                self.epochs_since_improvement = 0

    def adjust_learning_rate(self):
        old_lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= self.factor
        new_lr = self.optimizer.param_groups[0]['lr']
        self.lr_history.append(new_lr)
        previous_lr = self.lr_history[-2] if len(self.lr_history) > 1 else old_lr
        print(f'*** Learning rate changed from {previous_lr} to {new_lr} ***')

    def get_lr_history(self):
        return self.lr_history

def save_training_state(epoch, model, optimizer, lr_scheduler, best_loss, accuracy_history, loss_history, save_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
        'lr_scheduler': {
            'best_loss': lr_scheduler.best_loss,
            'epochs_since_improvement': lr_scheduler.epochs_since_improvement,
            'lr_history': lr_scheduler.lr_history
        },
        'accuracy_history': accuracy_history,
        'loss_history': loss_history
    }
    torch.save(state, save_path)

def load_training_state(load_path, model, optimizer, lr_scheduler):
    state = torch.load(load_path)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    lr_scheduler.best_loss = state['lr_scheduler']['best_loss']
    lr_scheduler.epochs_since_improvement = state['lr_scheduler']['epochs_since_improvement']
    lr_scheduler.lr_history = state['lr_scheduler']['lr_history']
    return state['epoch'], state['best_loss'], state['accuracy_history'], state['loss_history']

def main(rank, world_size, batch_size, num_epochs, state_path=None, cHead_path=None):
    setup(rank, world_size)
    print(state_path, cHead_path)
    dataset = load_dataset('imdb')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    full_dataset = concatenate_datasets([dataset['train'], dataset['test']])
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataset = IMDbDataset(train_dataset, tokenizer)
    test_dataset = IMDbDataset(test_dataset, tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)

    pretrained_model_name = 'roberta-large'
    roberta_model = RobertaModel.from_pretrained(pretrained_model_name)

    freeze_model_parameters(roberta_model)

    model = SentimentClassifier(roberta_model).to(rank)
    model = DDP(model, device_ids=[rank])

    if cHead_path:
        model.module.classifier.load_state_dict(torch.load(cHead_path))

    optimizer = AdamW(model.module.classifier.parameters(), lr=1e-5)
    lr_scheduler = CustomLRScheduler(optimizer)

    best_loss = float('inf')
    start_epoch = 0
    accuracy_history = []
    loss_history = []

    if state_path:
        print('state path given. Attempting to load...')
        start_epoch, best_loss, accuracy_history, loss_history = load_training_state(state_path, model, optimizer, lr_scheduler)
        print(f'{start_epoch} start epoch, {best_loss} best loss')

    # train loop
    for epoch in range(start_epoch, num_epochs):
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
        loss_history.append(avg_loss)
        if rank == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
        
        # constant lr this time
        #lr_scheduler.step(avg_loss)

        # eval on test set
        if avg_loss < best_loss:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc='Evaluating', position=rank):
                    input_ids = batch['input_ids'].to(rank)
                    attention_mask = batch['attention_mask'].to(rank)
                    labels = batch['labels'].to(rank)

                    outputs = model(input_ids, attention_mask=attention_mask)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accuracy_history.append(accuracy)

        if rank == 0:
            print(f'Test Accuracy: {accuracy:.2f}%')
            print(f'Average Loss = {avg_loss}')
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_classifier_head(model, 'imdb_classifier_head.pt')
                save_training_state(epoch + 1, model, optimizer, lr_scheduler, best_loss, accuracy_history, loss_history, 'training_state.pt')
                print(f'NEW BEST LOSS. MODEL SAVED.')

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=2, help='number of gpus')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size per gpu')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--state_path', type=str, default=None, help='path to training state')
    parser.add_argument('--cHead_path', type=str, default=None, help='path to saved classifier head')
    args = parser.parse_args()

    world_size = args.world_size
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    state_path = args.state_path
    cHead_path = args.cHead_path

    torch.multiprocessing.spawn(main, args=(world_size, batch_size, num_epochs, state_path, cHead_path), nprocs=world_size, join=True)

