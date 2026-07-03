# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, TransformerEncoder
from signjoey.decoders import Decoder, TransformerDecoder
from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params
from torch import Tensor
from typing import Union


class SignModel(nn.Module):
    def __init__(self, encoder, gloss_output_layer, decoder, sgn_embed, txt_embed,
                 gls_vocab, txt_vocab, do_recognition=True, do_translation=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed
        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab
        self.txt_bos_index = txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = txt_vocab.stoi[EOS_TOKEN]
        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    def forward(self, sgn, sgn_mask, sgn_lengths, txt_input, txt_mask=None):
        encoder_output, encoder_hidden = self.encode(sgn, sgn_mask, sgn_lengths)

        if self.do_recognition:
            gloss_scores = self.gloss_output_layer(encoder_output)
            gloss_probabilities = gloss_scores.log_softmax(2).permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output, encoder_hidden, sgn_mask, txt_input, unroll_steps, txt_mask=txt_mask
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities

    def encode(self, sgn, sgn_mask, sgn_length):
        embed_src = self.sgn_embed(x=sgn, mask=sgn_mask)
        encoder_output, encoder_hidden = self.encoder(embed_src, sgn_length, sgn_mask)
        return encoder_output, encoder_hidden

    def decode(self, encoder_output, encoder_hidden, sgn_mask, txt_input,
               unroll_steps, decoder_hidden=None, txt_mask=None):
        trg_embed = self.txt_embed(x=txt_input, mask=txt_mask)
        return self.decoder(
            trg_embed=trg_embed, encoder_output=encoder_output, encoder_hidden=encoder_hidden,
            src_mask=sgn_mask, trg_mask=txt_mask, unroll_steps=unroll_steps, hidden=decoder_hidden,
        )

    def get_loss_for_batch(self, batch: Batch, recognition_loss_function, translation_loss_function,
                            recognition_loss_weight, translation_loss_weight):
        decoder_outputs, gloss_probabilities = self.forward(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input, txt_mask=batch.txt_mask,
        )
        if self.do_recognition:
            recognition_loss = recognition_loss_function(
                gloss_probabilities, batch.gls, batch.sgn_lengths.long(), batch.gls_lengths.long()
            ) * recognition_loss_weight
        else:
            recognition_loss = None

        if self.do_translation:
            word_outputs, _, _, _ = decoder_outputs
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = translation_loss_function(txt_log_probs, batch.txt) * translation_loss_weight
        else:
            translation_loss = None

        return recognition_loss, translation_loss

    def __repr__(self):
        return "%s(encoder=%s, decoder=%s)" % (self.__class__.__name__, self.encoder, self.decoder)
    
    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn, sgn_mask=batch.sgn_mask, sgn_length=batch.sgn_lengths
        )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.sgn_mask,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(cfg: dict, sgn_dim: int, gls_vocab: GlossVocabulary,
                      txt_vocab: TextVocabulary, do_recognition=True, do_translation=True) -> SignModel:
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"], num_heads=cfg["encoder"]["num_heads"], input_size=sgn_dim,
    )
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)

    encoder = TransformerEncoder(
        **cfg["encoder"], emb_size=sgn_embed.embedding_dim, emb_dropout=enc_emb_dropout,
    )

    gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab)) if do_recognition else None

    txt_embed = Embeddings(
        **cfg["decoder"]["embeddings"], num_heads=cfg["decoder"]["num_heads"],
        vocab_size=len(txt_vocab), padding_idx=txt_padding_idx,
    )
    dec_dropout = cfg["decoder"].get("dropout", 0.0)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder = TransformerDecoder(
        **cfg["decoder"], vocab_size=len(txt_vocab), emb_size=txt_embed.embedding_dim,
        emb_dropout=dec_emb_dropout,
    )

    model = SignModel(
        encoder=encoder, gloss_output_layer=gloss_output_layer, decoder=decoder,
        sgn_embed=sgn_embed, txt_embed=txt_embed, gls_vocab=gls_vocab, txt_vocab=txt_vocab,
        do_recognition=do_recognition, do_translation=do_translation,
    )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    return model
