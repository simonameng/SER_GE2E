import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-e", "--enc_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt",
                        help="Path to a saved encoder")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("-s", "--source_audio_path", type=Path,
                        default="audio",
                        help="Path to the source audio folder")
    parser.add_argument("-d", "--dest_emb_path", type=Path,
                        default="embed",
                        help="Path to the destination embedding folder")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    print("Preparing the encoder, the synthesizer and the vocoder...")
    # ensure_default_models(Path("saved_models"))
    encoder.load_model(args.enc_model_fpath)


    ## Run a test
    print("Testing your configuration with small inputs.")
    # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # sampling rate, which may differ.
    # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # The sampling rate is the number of values (samples) recorded per second, it is set to
    # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # to an audio of 1 second.
    print("\tTesting the encoder...")
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    print("Encoder test passed!\n\n")

#------------------------------------------------------------------------------------------------------
    # in_fpath = "audio/Ses01F_impro01_F000.wav"
    # preprocessed_wav = encoder.preprocess_wav(in_fpath)
    # embed = encoder.embed_utterance(preprocessed_wav)
    # print("Embedding Content:\n", embed)
    source_dir = arg_dict.pop("source_audio_path")
    dest_dir = arg_dict.pop("dest_emb_path")

    if not(os.path.exists(dest_dir)):
                    os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        for D in dirs:
            src_path = os.path.join(source_dir, D, "wav")
            for r, d, f in os.walk(src_path):
                for file in f:
                    if file.endswith(".wav"):
                        wavpath = os.path.join(r, file)
                        preprocessed_wav = encoder.preprocess_wav(wavpath)
                        embed = encoder.embed_utterance(preprocessed_wav)
                        dest_emo_dir = os.path.join(dest_dir, D)
                        if not(os.path.exists(dest_emo_dir)):
                            os.makedirs(dest_emo_dir)
                        npypath = os.path.join(dest_emo_dir, file.replace(".wav", ".npy"))
                        np.save(npypath, embed)