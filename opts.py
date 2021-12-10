import os
import logging

from utils.utils import str2bool
import cfg

logger = logging.getLogger(__name__)
log = logger


def add_arguments(parser):
    ''' Takes a parser and add arguments to it '''

    src_choices = cfg.src_choices
    model_choices = cfg.MODEL_LIST
    dataset_choices = cfg.dataset_choices
    prepare_choices = cfg.PREPARE_ENGINES
    permutation_choices = cfg.permutation_choices

    parser.add("--num_workers", type=int, default=4, help="dataloader number of workers")
    parser.add("--jid", type=int, default=1, help="job id: 1[default] used for job queue")
    parser.add("--seed", type=int, default=150344, help="seed to use: 150344[default]")
    parser.add("--dataset", type=str, choices=dataset_choices, default="webnlg", help="dataset: webnlg[default]")
    parser.add("--subset_fraction", type=int, default=None, help="subsample the dataset. ['None' for webnlg; 50000 for tekgen] for training")
    parser.add("--prepare", type=str, choices=prepare_choices, default="webnlg", help="prepare engine: webnlg[default]")
    parser.add("--prepare_permutation", type=str, choices=permutation_choices, default="none", help="prepare permutation: 'none'[default]")
    parser.add("--model", type=str, choices=model_choices, default="t5-base", help="model: t5-base[default]")
    parser.add("--src", type=str, choices=src_choices, default='A', help="source: 'A'[default]|'B'|'A+B' if 'A', source is A (and target is B). if 'B', source is 'B' (and target is B), if 'A+B', source is [A|B] (and target is [B|A])")
    parser.add("--batch_size", type=int, default=32, help="batch size")
    parser.add("--batch_size_eval", type=int, default=20, help="batch size for evaluation")
    parser.add("--lr", type=float, default=1e-3, help="learning rate: 1e-3[default]")
    parser.add("--train_minib_aggregate", type=int, default=1, help='number of minib to aggregate in training')
    parser.add("--job_num_minib", type=int, default=-1, help="number of minibacth/iteration per job: '-1[default]|-1*E|N' where  N>0 is number of minibatches, '-1' means 1 epoch, '-2; means 2 epochs, etc.")
    parser.add("--checkpoint_every", type=int, default=-1, help="number of minibatch/iteration before checkpointing: '-1[default]|-1*E|N' where  N>0 is number of minibatches, '-1' means 1 epoch, '-2; means 2 epochs, etc.")
    parser.add("--log_every", type=int, default=50, help="Logging stats every N minibatches")
    parser.add("--valid_on_start", type=str2bool, default=False, help="evaluation on start, before training starts")
    parser.add("--valid_on_end", type=str2bool, default=False, help="evaluation on end, after training is done")
    parser.add("--valid_every", type=int, default=-1, help="evaluation run every N minibatches of training")
    parser.add("--valid_offline", type=str2bool, default=False, help="Perform *all* evaluations offline (aka checkpoints eval)")
    parser.add("--valid_num_minib", type=int, default=-1, help="Number of minibatches to use for validation")
    parser.add("--valid_beam_size", type=int, default=5, help="Beam size for generation on validation")
    parser.add("--valid_max_length", type=int, default=200, help="Max length for generation on validation")
    parser.add("--valid_generate_on_end", type=str2bool, default=False, help="evaluation on end, after training is done")
    parser.add("--train_samples_every", type=int, default=400, help="Display training samples every N minibatches 400[default]")
    parser.add("--timer_progress_every", type=int, default=500, help="show timer progress every N minibatch")
    parser.add("--output_dir", type=str, default="./output/", help='directory of output. tb, model checkpoints, logging, etc. are saved in subdirs')
    # SCST related args
    parser.add("--scst", type=str2bool, default=False, help='enable SCST training of model')
    parser.add("--scst_metrics", type=str, default='', help="SCST metrics to use e.g. 'exactF1:1.0' or 'bleu_nltk:1.0,meteor:0.9' following the format '<metric1>:<weight1>,<metric2>:<weight2>,...'")
    parser.add("--scst_checkpoint_id", type=int, default=-1, help="checkpoint state id number of the CE experiments checkpoints you want to start SCST from")
    parser.add("--scst_checkpoint_dir", type=str, default='', help="checkpoints dir for CE experiments SCST will start from")

    parser.add("--grad_clip", type=str2bool, default=False, help='gradient clipping')
    parser.add("--clip_max_norm", type=float, default=0.25, help='gradient clipping max norm (float): 0.25[default]')
    parser.add("--clip_norm_type", type=float, default=2.0, help='gradient clipping norm_type (float): 2.0[default|inf')

    return parser


def check_and_add_arguments(args):
    ''' Checks arguments and add derived arguments '''

    # add derived arguments
    args.log_dir = os.path.join(args.output_dir, "logs", f'{args.jid:02d}')
    args.tensorboard_dir = os.path.join(args.output_dir, "tb", f'{args.jid:02d}')
    args.json_logger_dir = os.path.join(args.output_dir, "jsonlog", f'{args.jid:02d}')
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    args.checkpoint_eval_dir = os.path.join(args.output_dir, "checkpoints_eval")

    # check arguments
    if args.num_workers <= 0:
        raise ValueError("Number of workers must be greater than 0")

    if args.prepare == 'webnlg-lex1-kbpm' and args.prepare_permutation != 'internal':
        raise ValueError(f"prepare engine {args.prepare} does triple permutation internally -> args.prepare_permutation = 'internal' must be set")

    if args.scst:

        if not args.scst_metrics:
            raise ValueError('Metrics to use for SCST must be defined')
        else:
            metrics_weights = args.scst_metrics.split(',')
            if len(metrics_weights) > 1:
                q, r = divmod(len(metrics_weights), 2)
                if r != 0:
                    raise ValueError("scst_metrics string must follow the format '<metric1>:<weight1>,<metric2>:<weight2>,...'")

            metric_src = {'exactF1': ['A', 'A+B'],
                          'sacre_bleu': ['B', 'A+B'],
                          'sacre_chrf': ['B', 'A+B'],
                          'sacre_ter': ['B', 'A+B'],
                          'bleu_nltk': ['B', 'A+B', 'A'],  # allow bleu_nltk for graph generation (experimental)
                          'meteor_nltk': ['B', 'A+B']}

            use_metric_weight = []

            args.webnlg_exact_F1_weight = None
            args.sacre_bleu_weight = None
            args.sacre_chrf_weight = None
            args.sacre_ter_weight = None
            args.bleu_nltk_weight = None
            args.meteor_nltk_weight = None

            for metric_weight in metrics_weights:

                metric, weight = metric_weight.split(':')
                wght = float(weight)

                # compatibility check: metric and args.src must agree...
                if args.src not in metric_src[metric]:
                    raise ValueError(f"scst metric '{metric}' is only for src '{metric_src[metric]} and not args.src='{args.src}'")

                if metric == 'exactF1':
                    args.webnlg_exact_F1_weight = wght
                elif metric == 'sacre_bleu':
                    args.sacre_bleu_weight = wght
                elif metric == 'sacre_chrf':
                    args.sacre_chrf_weight = wght
                elif metric == 'sacre_ter':
                    args.sacre_ter_weight = wght
                elif metric == 'bleu_nltk':
                    args.bleu_nltk_weight = wght
                elif metric == 'meteor_nltk':
                    args.meteor_nltk_weight = wght
                else:
                    raise ValueError(f"# scst metric '{metric}' is unknown")

                use_metric_weight.append((metric, wght))

            args.use_metric_weight = use_metric_weight

    return args


def add_arguments_generate(parser):
    ''' Takes a parser for generation and adds arguments to it'''

    dataset_choices = cfg.dataset_choices
    prepare_choices = cfg.PREPARE_ENGINES
    split_choices = cfg.split_logic_choices

    parser.add("--state_path", type=str, default='', help="path to state file")
    parser.add("--dataset", type=str, choices=dataset_choices, default="webnlg", help="dataset: 'webnlg[default]'")
    parser.add("--subset_fraction", type=int, default=None, help="subsample the dataset. ['None' for webnlg; 50000 for tekgen] for training, both None for testing")

    parser.add("--prepare", type=str, choices=prepare_choices, default="webnlg", help="prepare engine: webnlg[default]")
    parser.add("--split", type=str, choices=split_choices, default='', help="split: 'valA|valB|testA|testB [no default]'")
    parser.add("--num_workers", type=int, default=4, help="dataloader number of workers")
    parser.add("--batch_size_eval", type=int, default=20, help="batch size for evaluation")
    parser.add("--validate", type=str2bool, default=True, help="perform validation in evaluation 'True[default]|False'")
    parser.add("--generate", type=str2bool, default=True, help="perform generation in evaluation 'True[default]|False'")
    parser.add("--valid_beam_size", type=int, default=5, help="Beam size for generation on validation")
    parser.add("--valid_max_length", type=int, default=200, help="Max length for generation on validation")
    parser.add("--output_dir", type=str, default='./output_eval', help="output dir for evaluation")

    return parser


def check_and_add_arguments_generate(args):
    ''' Checks arguments and add derived arguments for generation args parser'''

    # add derived arguments
    args.distributed = False  # some code requires this (dataloaders, etc.)
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.tensorboard_dir = os.path.join(args.output_dir, "tb")
    args.json_logger_dir = os.path.join(args.output_dir, "jsonlog")  # machine readable logging
    args.generate_dir = os.path.join(args.output_dir, "generate")

    # checks
    if args.dataset not in cfg.GENERATION_DATASETS:
        raise ValueError(f"dataset '{args.dataset}' is not a dataset supported for generation: '{args.GENERATION_DATASETS}'")

    if args.prepare not in cfg.GENERATION_PREPARE:
        raise ValueError(f"prepare engine '{args.prepare}' is not supported for generation: '{args.GENERATION_PREPARE}'")

    return args
