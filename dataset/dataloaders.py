import logging
from types import SimpleNamespace

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from dataset.webnlg_dataset import WebNLGDataset
from dataset.tekgen_dataset import TekGenDataset
import cfg

logger = logging.getLogger(__name__)
log = logger


def get_dataloaders(args, tokenizer, split=None):
    ''' Returns a dict of dataloaders given args.dataset, tokenizer, and split'''

    splits = cfg.split_choices_all
    dataloaders = {}

    log.info(f"dataloader requested for: [{args.dataset}/{splits if split is None else split}]")

    if split is not None and split not in splits:
        raise ValueError(f"# split {split} is not in the list {splits}")

    split = splits if split is None else [split]

    if args.dataset in cfg.dataset_choices:

        for key in split:
            dl = get_dataloader(args, tokenizer, key)
            dataloaders[key] = dl

    else:
        raise RuntimeError(f"# dataset '{args.dataset}' is not valid")

    return dataloaders


def get_dataset(args, tokenizer, split):
    ''' Returns a dataset given args.dataset, args.prepare, tokenizer and split (and few options from args)
    Args:
       tokenizer: tokenizer
       split: dataset split

       from args: see {root}/opts.py for details
          args.dataset : dataset
          args.prepare : preparation engine
          args.prepare_permutation : permutation scheme for prepare engine
          args.src : source modality
    '''

    if args.dataset not in cfg.dataset_choices:
        raise RuntimeError(f"# dataset '{args.dataset}' is not valid")

    if args.dataset == 'webnlg':
        prepare_args = SimpleNamespace(prepare_permutation=args.prepare_permutation)
        ds = WebNLGDataset(split, tokenizer, args.src, prepare=args.prepare, prepare_args=prepare_args)
    elif args.dataset == 'tekgen':
        prepare_args = SimpleNamespace(prepare_permutation=args.prepare_permutation)
        ds = TekGenDataset(split, tokenizer, args.src, prepare=args.prepare, prepare_args=prepare_args)
    else:
        raise RuntimeError(f"# dataset '{args.dataset}' is not valid")

    return ds


def get_dataloader(args, tokenizer, split):
    ''' Gets a dataloader for a given split of a dataset, and tokenizer
    Args:
       tokenizer: tokenizer
       split: dataset split

      from args: see {root}/opts.py for details
         args.dataset : dataset
         args.subset_fraction : subset fraction of split to use
    '''

    is_train = True if split == 'train' else False
    dl, ds = None, None

    ds = get_dataset(args, tokenizer, split)

    # tekgen is restricted in numbers of samples
    if 'tekgen' in args.dataset:
        if args.subset_fraction is None:
            if 'test' in split:
                args.subset_fraction = 50000
            elif 'val' in split:
                args.subset_fraction = 5000
            else:
                raise ValueError(f"split {split} is not recognized for tekgen")
            log.info(f"# for dataset '{args.dataset}': restricting samples w/ args.subset_fraction={args.subset_fraction}")

    if args.subset_fraction is not None:
        subset_indices = list(range(args.subset_fraction))
        log.info(f'# use subset of dataset: [{len(subset_indices)}]')
        ds_main = ds
        ds = Subset(ds_main, subset_indices)
        ds._collate_fn = ds_main._collate_fn
        ds.tokenizer = ds_main.tokenizer
        ds.get_meta = ds_main.get_meta  # works bc we take the first args.subset_fraction samples from ds_main

    sampler = None

    if is_train:
        sampler = DistributedSampler(ds, num_replicas=args.world_size,
                                     rank=args.local_rank) if args.distributed else None

    dl = DataLoader(ds,
                    batch_size=args.batch_size if is_train else args.batch_size_eval,
                    shuffle=False,
                    collate_fn=ds._collate_fn,
                    num_workers=args.num_workers,
                    sampler=sampler,
                    drop_last=True if is_train else False)

    log.info(f"# dataloader: '{args.dataset}' {split} [{len(dl)}]  (dataset:{len(ds)} samples)")

    # check
    if len(dl) == 0:
        raise RuntimeError(f'dataloader for dataset split {split} is empty: len(dl)={len(dl)}')

    return dl
