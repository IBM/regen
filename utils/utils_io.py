import logging.handlers
import logging

import torch

from models.model import Text2KB2Text

logger = logging.getLogger(__name__)
log = logger


def load_model(args, vocab, device, state, no_random_init=True):
    ''' Loads model given a vocab, device, state
    '''
    model = Text2KB2Text(embedding_size=args.emb_dim,
                         hidden_size=args.hidden_size,
                         vocab=vocab,
                         pad_token_id=vocab.pad_token_id,
                         bidirectional_encoder=args.bidirectional_encoder,
                         attention=args.use_attention,
                         enc_type=args.enc_type,
                         dec_type=args.dec_type,
                         detach_in_soft=args.detach_in_soft,
                         transformer_dim_feedforw=args.transformer_dim_feedforw,
                         num_transformer_heads=args.num_transformer_heads,
                         num_transformer_layers=args.num_transformer_layers,
                         transformer_dropout=args.transformer_dropout,
                         device=device,
                         copy_mechanism=args.copy_mechanism,
                         model_dynamic_seqlen=args.model_dynamic_seqlen,
                         aggregate_num_minib=args.aggregate_num_minib,
                         decoding_method=args.decoding_method,
                         args=args)

    if (args.enc_type == 'bert') and (args.dataset == 'conceptnet.supervised.bert.strict'):
        log.info("# bert original vocab: {}".format(model.encoder.enc.embeddings.word_embeddings.weight.size()))
        model.encoder.reduce_vocab(vocab)
        log.info("# bert after reduce vocab: {}".format(model.encoder.enc.embeddings.word_embeddings.weight.size()))

        if args.freeze_enc:
            for param in model.encoder.enc.parameters():
                param.requires_grad = False

    model.to(device)

    # Load model from state
    log.info("# Model loading from state")
    if (state.get('paths', None) is not None) and ('model_path' in state['paths']):
        model.load_state_dict(torch.load(state['paths']['model_path'], map_location=device))  # maps to current device on the fly
        log.info('# Loaded model params from: {}'.format(state['paths']['model_path']))
    elif (state.get('paths', None) is not None) and ('model_path' not in state['paths']):
        raise Exception('missing model_path in state')
    else:
        if no_random_init:
            raise Exception('model cannot be initialized from a state_dict')
        else:
            log.warning('# Model initialized randomly')

    return model
