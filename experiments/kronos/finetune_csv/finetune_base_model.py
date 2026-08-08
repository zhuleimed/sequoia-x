import os
import sys
import json
import time
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig


class CustomKlineDataset(Dataset):
    
    def __init__(self, data_path, data_type='train', lookback_window=90, predict_window=10,
                 clip=5.0, seed=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                 stride=1):
        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        # ⚠️ 2026-08-08 扩量优化: 滑动窗口 stride（窗口重叠 140/141 冗余大,
        # stride=5 样本降 5 倍, 信息损失 <5%, 训练成本降 5 倍）;
        # 但序列起点偏移必须 < stride, 否则覆盖不全 → 每组内偏移以 stride 为步长,
        # 组内样本数 = ceil((g_len - window + 1) / stride)
        self.stride = max(1, stride)
        
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        self.py_rng = random.Random(seed)

        self._load_and_preprocess_data()
        self._split_data_by_time()

        # ⚠️ 2026-08-08 多股票支持（A 股微调）: 窗口必须落在同一 symbol 内,
        # 否则跨股票拼接序列产生伪窗口。n_samples = 各 symbol 组内样本数之和。
        self.group_offsets = []   # 每组 [start, end) 行区间
        if 'symbol' in self.symbols and self.symbols.nunique() > 1:
            cur_sym, cur_start = self.symbols.iloc[0], 0
            for i in range(1, len(self.symbols) + 1):
                if i == len(self.symbols) or self.symbols.iloc[i] != cur_sym:
                    g_len = i - cur_start
                    if g_len >= self.window:
                        self.group_offsets.append((cur_start, i))
                    cur_sym, cur_start = self.symbols.iloc[i] if i < len(self.symbols) else None, i
        else:
            self.group_offsets = [(0, len(self.data))]
        self.group_starts = [g[0] for g in self.group_offsets]
        self.group_lens = [g[1] - g[0] for g in self.group_offsets]
        self.group_samps = [max(0, (gl - self.window + self.stride) // self.stride)
                            for gl in self.group_lens]
        self.n_samples = sum(self.group_samps)

        print(f"[{data_type.upper()}] Data length: {len(self.data)}, Available samples: {self.n_samples} "
              f"(groups={len(self.group_offsets)})")

    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)

        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values(['symbol', 'timestamps']).reset_index(drop=True)

        self.timestamps = df['timestamps'].copy()
        self.symbols = df['symbol'].copy() if 'symbol' in df.columns else pd.Series([''] * len(df))

        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month

        self.data = df[self.feature_list + self.time_feature_list].copy()

        if self.data.isnull().any().any():
            print("Warning: Missing values found in data, performing forward fill")
            self.data = self.data.fillna(method='ffill')

        print(f"Original data time range: {self.timestamps.min()} to {self.timestamps.max()}")
        print(f"Original data total length: {len(df)} records, symbols: {self.symbols.nunique()}")

    def _split_data_by_time(self):
        total_length = len(self.data)

        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))

        # 多股票: 每 symbol 组内按时间比例切（各组时间轴一致, 组内前段=早期）
        if self.symbols.nunique() > 1:
            mask = pd.Series(False, index=self.data.index)
            for sym, g in self.symbols.groupby(self.symbols):
                gi = g.index  # 组在全局 df 中的 index
                n = len(g)
                if self.data_type == 'train':
                    mask.loc[gi[:int(n * self.train_ratio)]] = True
                elif self.data_type == 'val':
                    a = int(n * self.train_ratio)
                    b = int(n * (self.train_ratio + self.val_ratio))
                    mask.loc[gi[a:b]] = True
                else:
                    mask.loc[gi[int(n * (self.train_ratio + self.val_ratio)):]] = True
            self.data = self.data[mask].copy()
            self.timestamps = self.timestamps[mask].copy()
            self.symbols = self.symbols[mask].copy()
            print(f"[{self.data_type.upper()}] 按 symbol 组内比例切分 → {len(self.data)} records")
            return

        if self.data_type == 'train':
            self.data = self.data.iloc[:train_end].copy()
            self.timestamps = self.timestamps.iloc[:train_end].copy()
            print(f"[{self.data_type.upper()}] Training set: first {train_end} time points ({self.train_ratio})")
            print(f"[{self.data_type.upper()}] Training set time range: {self.timestamps.min()} to {self.timestamps.max()}")
        elif self.data_type == 'val':
            self.data = self.data.iloc[train_end:val_end].copy()
            self.timestamps = self.timestamps.iloc[train_end:val_end].copy()
            print(f"[{self.data_type.upper()}] Validation set: time points {train_end+1} to {val_end} ({self.val_ratio})")
            print(f"[{self.data_type.upper()}] Validation set time range: {self.timestamps.min()} to {self.timestamps.max()}")
        elif self.data_type == 'test':
            self.data = self.data.iloc[val_end:].copy()
            self.timestamps = self.timestamps.iloc[val_end:].copy()
            print(f"[{self.data_type.upper()}] Test set: after time point {val_end+1}")
            print(f"[{self.data_type.upper()}] Test set time range: {self.timestamps.min()} to {self.timestamps.max()}")

        print(f"[{self.data_type.upper()}] Data length after split: {len(self.data)} records")
    
    def set_epoch_seed(self, epoch):
        epoch_seed = self.seed + epoch
        self.py_rng.seed(epoch_seed)
        self.current_epoch = epoch
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # 多股票: 全局 idx → (组, 组内偏移×stride); 窗口保证在同一 symbol 内
        if len(self.group_offsets) > 1:
            g = 0
            while g < len(self.group_samps) - 1 and idx >= self.group_samps[g]:
                idx -= self.group_samps[g]
                g += 1
            if idx >= self.group_samps[g]:
                raise IndexError("index out of range")
            start_idx = self.group_starts[g] + idx * self.stride
        else:
            start_idx = idx * self.stride
            if start_idx > len(self.data) - self.window:
                raise IndexError("index out of range")

        if self.data_type == 'train':
            epoch = getattr(self, 'current_epoch', 0)
            if len(self.group_offsets) > 1:
                max_start = (self.group_lens[g] - self.window) // self.stride
                start_idx = self.group_starts[g] + ((idx * 9973 + (epoch + 1) * 104729)
                                                    % (max_start + 1)) * self.stride
            else:
                max_start = (len(self.data) - self.window) // self.stride
                start_idx = ((idx * 9973 + (epoch + 1) * 104729) % (max_start + 1)) * self.stride

        end_idx = start_idx + self.window

        window_data = self.data.iloc[start_idx:end_idx]

        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor




def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Basemodel Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")
    
    train_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type='train',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        stride=getattr(config, 'stride', 1)
    )

    val_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type='val',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        stride=getattr(config, 'stride', 1)
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def train_model(model, tokenizer, device, config, save_dir, logger):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.basemodel_epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float('inf')
    batch_idx_global = 0
    
    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, s1_loss, s2_loss = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {loss.item():.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
            
            batch_idx_global += 1
        
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                
                logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                
                val_loss += loss.item()
                val_batches += 1
        
        if use_ddp:
            tensor_sum = torch.tensor([epoch_train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_batches_all = int(tensor_sum[1].item())
            val_loss_all = tensor_sum[2].item()
            val_batches_all = int(tensor_sum[3].item())
            avg_train_loss = (epoch_train_loss_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_batches_all) if val_batches_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
                       f"Training Loss: {avg_train_loss:.4f}\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Epoch Time: {epoch_time:.2f} seconds\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)
    
    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Basemodel Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = CustomFinetuneConfig(args.config)
    
    os.makedirs(config.basemodel_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    logger.info("Loading pretrained model or random initialization...")
    print("Loading pretrained model or random initialization...")
    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer = KronosTokenizer.from_pretrained(config.finetuned_tokenizer_path)
    else:
        import json, os
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture for training")
        cfg_path_tok = os.path.join(config.pretrained_tokenizer_path if hasattr(config, 'pretrained_tokenizer_path') else config.finetuned_tokenizer_path, 'config.json')
        with open(cfg_path_tok, 'r') as f:
            arch_t = json.load(f)
        tokenizer = KronosTokenizer(
            d_in=arch_t.get('d_in', 6),
            d_model=arch_t.get('d_model', 256),
            n_heads=arch_t.get('n_heads', 4),
            ff_dim=arch_t.get('ff_dim', 512),
            n_enc_layers=arch_t.get('n_enc_layers', 4),
            n_dec_layers=arch_t.get('n_dec_layers', 4),
            ffn_dropout_p=arch_t.get('ffn_dropout_p', 0.0),
            attn_dropout_p=arch_t.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch_t.get('resid_dropout_p', 0.0),
            s1_bits=arch_t.get('s1_bits', 10),
            s2_bits=arch_t.get('s2_bits', 10),
            beta=arch_t.get('beta', 0.05),
            gamma0=arch_t.get('gamma0', 1.0),
            gamma=arch_t.get('gamma', 1.1),
            zeta=arch_t.get('zeta', 0.05),
            group_size=arch_t.get('group_size', 4)
        )

    if getattr(config, 'pre_trained_predictor', True):
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
    else:
        import json, os
        print("pre_trained_predictor=False, randomly initializing Predictor architecture for training")
        cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
        model = Kronos(
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            n_layers=arch.get('n_layers', 12),
            d_model=arch.get('d_model', 832),
            n_heads=arch.get('n_heads', 16),
            ff_dim=arch.get('ff_dim', 2048),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.2),
            token_dropout_p=arch.get('token_dropout_p', 0.0),
            learn_te=arch.get('learn_te', True)
        )
    
    tokenizer = tokenizer.to(device)
    model = model.to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_size:,}")
    print(f"Model parameters: {model_size:,}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.predictor_learning_rate}")
    logger.info(f"Training epochs: {config.basemodel_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Tokenizer path: {config.finetuned_tokenizer_path}")
    logger.info(f"Pretrained model path: {config.pretrained_predictor_path}")
    
    logger.info("Starting fine-tuning training...")
    print("Starting fine-tuning training...")
    best_val_loss = train_model(model, tokenizer, device, config, config.basemodel_save_path, logger)
    
    final_msg = f"Training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.basemodel_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
