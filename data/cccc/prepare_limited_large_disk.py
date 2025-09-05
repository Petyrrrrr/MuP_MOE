import os

# IMPORTANT: Set cache directories to use the large disk before importing datasets
os.environ['HF_HOME'] = '/mnt/local/huggingface_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/mnt/local/huggingface_cache/hub'
os.environ['HF_DATASETS_CACHE'] = '/mnt/local/huggingface_cache/datasets'

from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import argparse

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def estimate_tokens_per_example(dataset, sample_size=1000):
    """Estimate average tokens per example by sampling the dataset"""
    actual_sample_size = min(sample_size, len(dataset))
    sample_indices = np.random.choice(len(dataset), actual_sample_size, replace=False)
    
    total_tokens = 0
    for idx in tqdm(sample_indices, desc="Sampling for token estimation"):
        example = dataset[int(idx)]
        tokens = enc.encode_ordinary(example['text'])
        total_tokens += len(tokens)
    
    return total_tokens / actual_sample_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare C4 dataset with optional token limit')
    parser.add_argument('--max_tokens_billions', type=float, default=10.0,
                        help='Maximum number of tokens to process in billions (default: 10.0)')
    parser.add_argument('--no_limit', action='store_true',
                        help='Process the entire dataset without token limit')
    parser.add_argument('--dataset_fraction', type=float, default=0.25,
                        help='Fraction of dataset to download (0.0-1.0, default: 1.0 for full dataset)')
    args = parser.parse_args()
    
    print(f"Using cache directory: {os.environ['HF_DATASETS_CACHE']}")
    print(f"Available space on /mnt/local: {os.statvfs('/mnt/local').f_bavail * os.statvfs('/mnt/local').f_frsize / (1024**4):.2f} TB")
    
    print(f"\nLoading C4 dataset...")
    
    # Use streaming to limit download if fraction < 1.0
    if args.dataset_fraction < 1.0:
        print(f"Using streaming mode to download only {args.dataset_fraction*100:.1f}% of the dataset")
        
        # Load with streaming
        dataset_stream = load_dataset("c4", "en", streaming=True)
        
        # Estimate total size (C4 has ~365M examples in train)
        estimated_train_size = 365_000_000
        estimated_val_size = 365_000
        
        train_limit = int(estimated_train_size * args.dataset_fraction)
        val_limit = int(estimated_val_size * args.dataset_fraction)
        
        print(f"Taking first {train_limit:,} train examples and {val_limit:,} validation examples")
        
        # Take limited samples and convert to regular dataset
        train_data = list(tqdm(dataset_stream['train'].take(train_limit), 
                              total=train_limit, 
                              desc="Downloading train data"))
        val_data = list(tqdm(dataset_stream['validation'].take(val_limit),
                           total=val_limit,
                           desc="Downloading validation data"))
        
        # Convert to Dataset format
        from datasets import Dataset
        dataset = {
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data)
        }
    else:
        # Load full dataset as before
        dataset = load_dataset("c4", "en", num_proc=num_proc_load_dataset)
        dataset = {'train': dataset['train'], 'validation': dataset['validation']}
    
    # Print dataset info
    print(f"Total number of examples in train: {len(dataset['train'])}")
    print(f"Total number of examples in validation: {len(dataset['validation'])}")
    
    # Estimate tokens per example
    print("\nEstimating token count...")
    avg_tokens_train = estimate_tokens_per_example(dataset['train'])
    estimated_total_tokens = avg_tokens_train * len(dataset['train'])
    
    print(f"\nAverage tokens per example: {avg_tokens_train:.2f}")
    print(f"Estimated total tokens in train set: {estimated_total_tokens:,.0f}")
    print(f"Estimated total tokens in billions: {estimated_total_tokens / 1e9:.2f}B")
    
    # Determine how many examples to use
    if args.no_limit:
        max_examples = len(dataset['train'])
        print(f"\nProcessing entire dataset...")
    else:
        max_tokens = args.max_tokens_billions * 1e9
        max_examples = int(max_tokens / avg_tokens_train)
        max_examples = min(max_examples, len(dataset['train']))
        
        estimated_tokens_to_process = max_examples * avg_tokens_train
        print(f"\nLimiting to approximately {args.max_tokens_billions}B tokens")
        print(f"Will process {max_examples:,} examples (out of {len(dataset['train']):,})")
        print(f"Estimated tokens to process: {estimated_tokens_to_process:,.0f} ({estimated_tokens_to_process/1e9:.2f}B)")
    
    # Create limited dataset if needed
    if max_examples < len(dataset['train']):
        print("\nCreating limited dataset...")
        # Shuffle and select first N examples
        train_indices = np.arange(len(dataset['train']))
        np.random.shuffle(train_indices)
        selected_indices = train_indices[:max_examples]
        
        limited_train = dataset['train'].select(selected_indices)
        
        # Update dataset with limited train
        dataset['train'] = limited_train
    
    # Split dataset
    split_dataset = dataset['train'].train_test_split(test_size=0.001, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f"\n{split} split: {len(dset)} examples, {arr_len:,} tokens ({arr_len/1e9:.2f}B)")
        
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        
    # Print final summary
    print("\n=== Final Token Count Summary ===")
    total_tokens = 0
    for split in tokenized.keys():
        token_count = np.sum(tokenized[split]['len'], dtype=np.uint64)
        total_tokens += token_count
        print(f"{split}: {token_count:,} tokens ({token_count/1e9:.2f}B)")
    print(f"Total: {total_tokens:,} tokens ({total_tokens/1e9:.2f}B)")
