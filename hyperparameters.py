# Copyright 2023 Vikaas Varma
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Hyperparameters for various algorithms in the codebase

# Transformer hyperparameters

MAX_SEQ_LEN = 1024
NUM_LAYERS = 6
NUM_HEADS = 8
DIM_FF = 512
ATTENTION_HIDDEN_SIZE = 384


# Training hyperparameters

BATCH_SIZE = 256
MINI_BATCH_SIZE = 32  # Take into account system memory
assert BATCH_SIZE % MINI_BATCH_SIZE == 0

DROPOUT_PROB = 0.15
LEARNING_RATE = 0.1
NUM_WARMUP_STEPS = 8000
RSQRT_DECAY = True

DEFAULT_CONFIG = {
    "learning_rate": LEARNING_RATE,
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "dim_ff": DIM_FF,
    "attention_hidden_size": ATTENTION_HIDDEN_SIZE,
}


# MIDI processing
SAMPLES_PER_BEAT = 4


# Melody inference

# Parameters for denoising and extracting pitch contours
# http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamongomezmelodytaslp2012.pdf
THRESHOLD_FACTOR = 0.9  # Ratio of max saliency to threshold
THRESHOLD_DEVIATION = 0.9  # Number of standard deviations below mean to threshold
VOICING_THRESHOLD_DEVIATION = 0.2  # Threshold for voicing detection
