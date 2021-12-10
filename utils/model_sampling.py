""" Model sampling

The code below was derived from

https://github.com/huggingface/transformers/blob/v4.2.2/src/transformers/generation_utils.py

which is under Apache 2.0 License (https://www.apache.org/licenses/LICENSE-2.0)

The goal is to have access of the internals of the generate() method to modify sampling at inference time.
The code is deeply connected to the API used in package transformers (4.2.2) by HuggingFace
and may not work foor other library version.

"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from transformers.file_utils import ModelOutput

from transformers.modeling_outputs import (
    BaseModelOutput,
    # Seq2SeqLMOutput,  # -> we are using a custom SamplingSeq2SeqLMOutput instead
)

from transformers.generation_utils import (
    SampleEncoderDecoderOutput,
    # SampleDecoderOnlyOutput  # -> narrow to encoder-decoder case only
)


class T5Sampling(nn.Module):

    def __init__(self, args, model):
        '''
        Sampling of T5 model (conditional generation)
        '''
        super().__init__()
        self.args = args
        self.model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_sample=None
    ):
        r'''
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        '''

        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model.model_parallel:
            torch.cuda.set_device(self.model.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.model._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model.model_parallel:
            torch.cuda.set_device(self.model.decoder.first_device)
            hidden_states = hidden_states.to(self.model.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.model.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.model.decoder.first_device)

        # Decode
        if not do_sample:
            decoder_outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # decoder outputs:
            # BaseModelOutputWithPastAndCrossAttentions(
            #   last_hidden_state=hidden_states,
            #   past_key_values=present_key_value_states,
            #   hidden_states=all_hidden_states,
            #   attentions=all_attentions,
            #   cross_attentions=all_cross_attentions,
            # )

            # compute logits
            sequence_output = decoder_outputs[0]  # [bs x seq_len x hidden_size]  hidden_size==config.d_model

            # Set device for model parallelism
            if self.model.model_parallel:
                torch.cuda.set_device(self.model.encoder.first_device)
                self.model.lm_head = self.model.lm_head.to(self.model.encoder.first_device)
                sequence_output = sequence_output.to(self.model.lm_head.weight.device)

            if self.model.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model.model_dim ** -0.5)

            lm_logits = self.model.lm_head(sequence_output)  # bsize x seq_len x vocab_size

        else:

            sampling_kwargs = {
                'repetition_penalty': 1.0,
                'no_repeat_ngram_size': 0,
                'bad_words_ids': None,
                'max_length': decoder_input_ids.shape[-1],  # [bs x seq_len]
                'min_length': 10,
                'eos_token_id': self.model.config.eos_token_id,
                'pad_token_id': self.model.config.pad_token_id,
                'prefix_allowed_tokens_fn': None,
                'num_beams': 1,
                'num_beam_groups': 1,
                'diversity_penalty': 0.0,
                'top_k': 50,
                'top_p': 1.0,
                'temperature': 1.0,
                'num_beams': 1,
                'output_scores': True
            }

            sampling_outputs = self.sampling(
                encoder_outputs=encoder_outputs,  # save pointer to encoder outputs from source_ids
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **sampling_kwargs
            )
            # sampling outputs:
            # MySampleEncoderDecoderOutput(
            #   sequences=sample_input_ids,
            #   scores=scores, (logits from decoder call)
            #   encoder_attentions=encoder_attentions,
            #   encoder_hidden_states=encoder_hidden_states,
            #   decoder_attentions=decoder_attentions,
            #   decoder_hidden_states=decoder_hidden_states,
            #   sample_logprobs=sample_logprobs (logprobs for each sampled tokens (best token))
            # )
            lm_logits_tuple = sampling_outputs.scores  # tuple of modified logits:  ([bsize x vocab_size], ... (max_seq_len times))
            lm_logits_sampling = torch.stack(lm_logits_tuple, dim=1)  # bsize x max_seq_len x vocab_size -> max_seq_len != labels seq_len...

            # get sampled logits
            best_logprobs_tuple = sampling_outputs.sample_logprobs
            best_logprobs_sampling = torch.stack(best_logprobs_tuple, dim=1)

            bsize, sampling_seq_len, vocab_size = lm_logits_sampling.shape
            seq_len = labels.shape[-1]
            # adjust to labels seq_len for loss computation below
            if sampling_seq_len < seq_len:
                lm_logits = lm_logits_sampling.new_zeros(bsize, seq_len, vocab_size)
                lm_logits[:, :sampling_seq_len, :] = lm_logits_sampling.clone()
            elif sampling_seq_len == seq_len:
                lm_logits = lm_logits_sampling
            else:
                raise RuntimeError(f'# sampling sequence length [{sampling_seq_len}] is greater than labels [{seq_len}] -> adjust max_length when calling sampling()')

        # compute loss if given labels...
        # replace pad_token_id w/ ignore_index to ensure proper loss computation
        if labels is not None:
            pad_token_id = self.model.config.pad_token_id
            labels[labels == pad_token_id] = -100

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not do_sample:
            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
                return ((loss,) + output) if loss is not None else output

            return SamplingSeq2SeqLMOutput(
                sequences=None,  # should have a sequence of tokens (from decoder)
                loss=loss,
                logits=lm_logits,
                sample_logprobs=None,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        else:  # sampling

            # remove decoder_start_token_id from output_sequences
            sampling_seq = sampling_outputs.sequences[:, 1:]
            sampling_outputs.sequences = sampling_seq

            # shrink logit to the same max sequence length as for output_seq above
            lm_logits = lm_logits[:, :sampling_seq.shape[-1]]  # [bsize x max_sampling_seq_length]

            return SamplingSeq2SeqLMOutput(
                sequences=sampling_outputs.sequences,
                loss=loss,
                logits=lm_logits,
                sample_logprobs=best_logprobs_sampling,
                past_key_values=None,  # decoder_outputs.past_key_values  # could bring it from decoder in sampling() -- not now
                decoder_hidden_states=sampling_outputs.decoder_hidden_states,
                decoder_attentions=sampling_outputs.decoder_attentions,
                cross_attentions=None,  # decoder_outputs.cross_attentions, # coulf bring it from decoder in sampling() -- not now
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,  # from encoder_outputs
                encoder_hidden_states=sampling_outputs.encoder_hidden_states,
                encoder_attentions=sampling_outputs.encoder_attentions,
            )

    def sampling(
            self,
            encoder_outputs=None,
            input_ids=None,
            attention_mask=None,
            inputs_embeds=None,
            past_key_values=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **sampling_kwargs,
    ):

        model_kwargs = {
            'encoder_outputs': encoder_outputs,
            # 'input_ids': input_ids,
            'attention_mask': attention_mask,
            'inputs_embeds': inputs_embeds,
            'past_key_values': past_key_values,
            'encoder_hidden_states': encoder_hidden_states,
            'encoder_attention_mask': encoder_attention_mask,
            'head_mask': head_mask,
            'use_cache': use_cache,
            'output_attentions': output_attentions,
            'output_hidden_states': output_hidden_states,
            'return_dict': return_dict,
        }

        logits_proc_kwargs = {
            'repetition_penalty': sampling_kwargs.get('repetition_penalty', None),
            'no_repeat_ngram_size': sampling_kwargs.get('no_repeat_ngram_size', None),
            'bad_words_ids': sampling_kwargs.get('bad_words_ids', None),
            'min_length': sampling_kwargs.get('min_length', None),
            'eos_token_id': sampling_kwargs.get('eos_token_id', None),
            'prefix_allowed_tokens_fn': sampling_kwargs.get('prefix_allowed_tokens_fn', None),
            'num_beams': sampling_kwargs.get('num_beams', None),
            'num_beam_groups': sampling_kwargs.get('num_beam_groups', None),
            'diversity_penalty': sampling_kwargs.get('diversity_penalty', None)
        }

        logits_warper_kwargs = {
            'top_k': sampling_kwargs.get('top_k', None),
            'top_p': sampling_kwargs.get('top_p', None),
            'temperature': sampling_kwargs.get('temperature', None),
            'num_beams': sampling_kwargs.get('num_beams', None)
        }

        max_length = sampling_kwargs.get('max_length', input_ids.shape[1])
        eos_token_id = sampling_kwargs.get('eos_token_id', self.model.config.eos_token_id)
        pad_token_id = sampling_kwargs.get('pad_token_id', self.model.config.pad_token_id)

        # get distribution pre_processing samplers
        logits_processor = self.model._get_logits_processor(**logits_proc_kwargs)

        # get probability distribution warper
        logits_warper = self.model._get_logits_warper(**logits_warper_kwargs)

        output_scores = sampling_kwargs.get('output_scores', False)
        scores = () if (return_dict and output_scores) else None
        decoder_attentions = () if (return_dict and output_attentions) else None
        decoder_hidden_states = () if (return_dict and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # prepare first input_ids
        decoder_start_token_id = self.model.config.decoder_start_token_id
        sample_input_ids = input_ids.new_full(
            (input_ids.shape[0], 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device)

        # init sequence length tensors
        sequence_lengths, unfinished_sequences, cur_len = self.model._init_sequence_length_for_generation(
            sample_input_ids, max_length
        )
        # sequence_lengths: tensor of sequence length to generate
        # unfinished_sequences: tensor of unfinished seq (set ot 1 for all minib at first)

        # sampled token logprobs
        sample_logprobs = ()

        while cur_len < max_length:

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_sampling(sample_input_ids, **model_kwargs)
            # For T5:
            # {
            #  "decoder_input_ids": sample_input_ids,
            #  "past_key_values": past,
            #  "encoder_outputs": encoder_outputs,
            #  "attention_mask": attention_mask,
            #  "use_cache": use_cache
            # }

            # forward pass to get next token
            # calling the original forward pass
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # outputs:
            # Seq2SeqLMOutput(
            #   loss=loss,
            #   logits=lm_logits,
            #   past_key_values=decoder_outputs.past_key_values,
            #   decoder_hidden_states=decoder_outputs.hidden_states,
            #   decoder_attentions=decoder_outputs.attentions,
            #   cross_attentions=decoder_outputs.cross_attentions,
            #   encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            #   encoder_hidden_states=encoder_outputs.hidden_states,
            #   encoder_attentions=encoder_outputs.attentions,
            # )
            next_token_logits = outputs.logits[:, -1, :]  # [bs,vocab_size]

            # pre-process distribution
            # next_token_scores = logits_processor(sample_input_ids, next_token_logits)  # next_token_scores: [bs, vocab_size]
            # next_token_scores = logits_warper(sample_input_ids, next_token_scores)
            # not sure the steps above are differentiable...
            # they generate -inf for some entries as it is usually used to eliminate some samples in the softmax
            # -> not good as the loss computation will give inf then...

            next_token_scores = next_token_logits  # debug: for now just a pass through to see if the loss is proper

            # Store scores, attentions and hidden_states when required
            if return_dict:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = F.softmax(next_token_scores, dim=-1)  # [bs, vocab_size]
            logprobs = F.log_softmax(next_token_scores, dim=-1)  # [bs, vocab_size]

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # select score of next_tokens
            sample_logprobs += (logprobs.gather(1, next_tokens.view(-1, 1)), )

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            sample_input_ids = torch.cat([sample_input_ids, next_tokens[:, None]], dim=-1)
            cur_len = cur_len + 1

            # update sequence length
            if eos_token_id is not None:
                sequence_lengths, unfinished_sequences = self.model._update_seq_length_for_generation(
                    sequence_lengths, unfinished_sequences, cur_len, next_tokens == eos_token_id
                )

            # stop when there is a </s> in each sentence, or if we exceed the maximum length
            if unfinished_sequences.max() == 0:
                break

            # update model kwargs
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )

        if return_dict:
            if self.model.config.is_encoder_decoder:
                return MySampleEncoderDecoderOutput(
                    sequences=sample_input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    sample_logprobs=sample_logprobs
                )
            else:
                raise RuntimeError('# we only allow encoder+decoder models')
                # return SampleDecoderOnlyOutput(
                #     sequences=sample_input_ids,
                #     scores=scores,
                #     attentions=decoder_attentions,
                #     hidden_states=decoder_hidden_states,
                # )
        else:
            return sample_input_ids

    def prepare_inputs_for_sampling(
            self, sample_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            sample_input_ids = sample_input_ids[:, -1:]

        return {
            "decoder_input_ids": sample_input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": None,
            "use_cache": use_cache
        }


@dataclass
class MySampleEncoderDecoderOutput(SampleEncoderDecoderOutput):
    sample_logprobs: torch.FloatTensor = None


@dataclass
class SamplingSeq2SeqLMOutput(ModelOutput):
    '''
    Based on Seq2SeqLMOutput, this is a custom class for allowing a sequence in the output...
    Base class for sequence-to-sequence language models outputs.

    Args:
        sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.

        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Language modeling loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    '''
    sequences: torch.LongTensor = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    sample_logprobs: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
