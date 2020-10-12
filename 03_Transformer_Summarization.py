# Goal: Transformer_Summarization Function to generate summary of a news article
# This uses  transformer decoder using trax library 

import sys
import os
import time 
import numpy as np
import pandas as pd
import gin

import textwrap 
import trax 
from trax import layers as tl
from trax.fastmath import numpy as tnp 
np.set_printoptions(threshold=sys.maxsize)

# Step1: Create Positional Encoding
def createPosEncoder(vocabSize, embeddingDepth, dropout, maxLength, mode):
    return [ 
        tl.Embedding(vocabSize, embeddingDepthembeddingDepth),  
        tl.Dropout(rate=dropout, mode=mode), 
        tl.PositionalEncoding(max_len=maxLength, mode=mode)] 

# Step2: Create a Feed-forward layer
def createFeedForwardLayer(embeddingDepth, depth, dropout, mode, ffActivation):
    return [ 
        tl.LayerNorm(), 
        tl.Dense(depth), 
        ff_activation(),  # Generally ReLU
        tl.Dropout(rate=dropout, mode=mode), 
        tl.Dense(embeddingDepth), 
        tl.Dropout(rate=dropout, mode=mode) 
    ]

# Step3: Decoder block
def DecoderBlock(embeddingDepth, depth, n_heads,
                 dropout, mode, ffActivationffActivation):
    return [
      tl.Residual(
          tl.LayerNorm(), 
          tl.CausalAttention(d_feature, n_heads=n_heads, dropout=dropout, mode=mode) 
        ),
      tl.Residual(
          FeedForward(embeddingDepth, depth, dropout, mode, ffActivation)
        ),
      ]
# Step4: Transformer decoder
def TransformerLM(vocab_size=33300,
                  embeddingDepth=512,
                  depth=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  maxLength=4096,
                  mode='train',
                  ffActivation=tl.Relu):
    # Create stack (list) of decoder blocks with n_layers with necessary parameters
    decoder_blocks = [ 
        DecoderBlock(embeddingDepth, depth, n_heads, dropout, mode, ffActivation) for _ in range(n_layers)] 

    # Create the complete model as written in the figure
    return tl.Serial(
        tl.ShiftRight(mode=mode), 
        PositionalEncoder(vocab_size, embeddingDepth, dropout, maxLength, mode),
        decoder_blocks, 
        tl.LayerNorm(), 
        tl.Dense(vocab_size), 
        tl.LogSoftmax() 
    )
