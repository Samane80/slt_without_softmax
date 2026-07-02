# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from signjoey.decoders import Decoder, RecurrentDecoder, TransformerDecoder
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
