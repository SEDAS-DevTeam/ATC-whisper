#
# Sources:
#
# Conversion to pt: https://github.com/ggerganov/whisper.cpp/blob/master/models/ggml_to_pt.py
# Conversion to ggml: https://github.com/ggerganov/whisper.cpp/blob/master/models/convert-pt-to-ggml.py
# Conversion st to pt: https://github.com/openai/whisper/discussions/1525

# imports
import struct
import torch
import numpy as np
from collections import OrderedDict
from whisper import Whisper, ModelDimensions
import io
import os
import sys
import json
import base64
import safetensors.torch
from transformers import pipeline
import re
import whisper
from transformers import WhisperForConditionalGeneration
from pathlib import Path


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def ggml_to_pt(fname_inp, fname_out):
    # Open the ggml file
    with open(fname_inp, "rb") as f:
        # Read magic number and hyperparameters
        magic_number, n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer, n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, use_f16 = struct.unpack("12i", f.read(48))
        print(f"Magic number: {magic_number}")
        print(f"Vocab size: {n_vocab}")
        print(f"Audio context size: {n_audio_ctx}")
        print(f"Audio state size: {n_audio_state}")
        print(f"Audio head size: {n_audio_head}")
        print(f"Audio layer size: {n_audio_layer}")
        print(f"Text context size: {n_text_ctx}")
        print(f"Text head size: {n_text_head}")
        print(f"Mel size: {n_mels}")
        # Read mel filters
        # mel_filters = np.fromfile(f, dtype=np.float32, count=n_mels * 2).reshape(n_mels, 2)
        # print(f"Mel filters: {mel_filters}")
        filters_shape_0 = struct.unpack("i", f.read(4))[0]
        print(f"Filters shape 0: {filters_shape_0}")
        filters_shape_1 = struct.unpack("i", f.read(4))[0]
        print(f"Filters shape 1: {filters_shape_1}")

        # Read tokenizer tokens
        # bytes = f.read(4)
        # print(bytes)

        # for i in range(filters.shape[0]):
        # for j in range(filters.shape[1]):
        #     fout.write(struct.pack("f", filters[i][j]))
        mel_filters = np.zeros((filters_shape_0, filters_shape_1))

        for i in range(filters_shape_0):
            for j in range(filters_shape_1):
                mel_filters[i][j] = struct.unpack("f", f.read(4))[0]

        bytes_data = f.read(4)
        num_tokens = struct.unpack("i", bytes_data)[0]
        tokens = {}

        for _ in range(num_tokens):
            token_len = struct.unpack("i", f.read(4))[0]
            token = f.read(token_len)
            tokens[token] = {}

        # Read model variables
        model_state_dict = OrderedDict()
        while True:
            try:
                n_dims, name_length, ftype = struct.unpack("iii", f.read(12))
            except struct.error:
                break  # End of file
            dims = [struct.unpack("i", f.read(4))[0] for _ in range(n_dims)]
            dims = dims[::-1]
            name = f.read(name_length).decode("utf-8")
            if ftype == 1:  # f16
                data = np.fromfile(f, dtype=np.float16, count=np.prod(dims)).reshape(dims)
            else:  # f32
                data = np.fromfile(f, dtype=np.float32, count=np.prod(dims)).reshape(dims)

            if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
                data = data[:, 0]

            model_state_dict[name] = torch.from_numpy(data)

    # Now you have the model's state_dict stored in model_state_dict
    # You can load this state_dict into a model with the same architecture

    # dims = ModelDimensions(**checkpoint["dims"])
    # model = Whisper(dims)
    dims = ModelDimensions(
        n_mels=n_mels,
        n_audio_ctx=n_audio_ctx,
        n_audio_state=n_audio_state,
        n_audio_head=n_audio_head,
        n_audio_layer=n_audio_layer,
        n_text_ctx=n_text_ctx,
        n_text_state=n_text_state,
        n_text_head=n_text_head,
        n_text_layer=n_text_layer,
        n_vocab=n_vocab,
    )
    model = Whisper(dims)  # Replace with your model's class
    model.load_state_dict(model_state_dict)

    # Save the model in PyTorch format
    torch.save(model.state_dict(), fname_out)


def pt_to_ggml(fname_inp, dir_whisper, fname_out):
    dir_whisper = Path(dir_whisper)
    fname_out = Path(fname_out)

    def bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a signficant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    # try to load PyTorch binary data
    try:
        model_bytes = open(fname_inp, "rb").read()
        with io.BytesIO(model_bytes) as fp:
            checkpoint = torch.load(fp, map_location="cpu")
    except Exception:
        print("Error: failed to load PyTorch model file:", fname_inp)
        sys.exit(1)

    hparams = checkpoint["dims"]
    print("hparams:", hparams)

    list_vars = checkpoint["model_state_dict"]

    # load mel filters
    n_mels = hparams["n_mels"]
    with np.load(dir_whisper / "whisper" / "assets" / "mel_filters.npz") as f:
        filters = torch.from_numpy(f[f"mel_{n_mels}"])
        #print(filters)

    #code.interact(local=locals())

    # load tokenizer
    # for backwards compatibility, also check for older hf_transformers format tokenizer files
    # old format: dir_whisper/whisper/assets/[multilingual/gpt2]/vocab.json
    # new format: dir_whisper/whisper/assets/[multilingual/gpt2].tiktoken
    multilingual = hparams["n_vocab"] >= 51865
    tokenizer = dir_whisper / "whisper" / "assets" / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
    tokenizer_type = "tiktoken"
    if not tokenizer.is_file():
        tokenizer = dir_whisper / "whisper" / "assets" / (multilingual and "multilingual" or "gpt2") / "vocab.json"
        tokenizer_type = "hf_transformers"
        if not tokenizer.is_file():
            print("Error: failed to find either tiktoken or hf_transformers tokenizer file:", tokenizer)
            sys.exit(1)

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    if tokenizer_type == "tiktoken":
        with open(tokenizer, "rb") as f:
            contents = f.read()
            tokens = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)}
    elif tokenizer_type == "hf_transformers":
        with open(tokenizer, "r", encoding="utf8") as f:
            _tokens_raw = json.load(f)
            if '<|endoftext|>' in _tokens_raw:
                # ensures exact same model as tokenizer_type == tiktoken
                # details: https://github.com/ggerganov/whisper.cpp/pull/725
                del _tokens_raw['<|endoftext|>']
            tokens = {bytes([byte_decoder[c] for c in token]): int(idx) for token, idx in _tokens_raw.items()}

    # use 16-bit or 32-bit floats
    use_f16 = True
    if len(sys.argv) > 4:
        use_f16 = False

    fout = fname_out.open("wb")

    fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
    fout.write(struct.pack("i", hparams["n_vocab"]))
    fout.write(struct.pack("i", hparams["n_audio_ctx"]))
    fout.write(struct.pack("i", hparams["n_audio_state"]))
    fout.write(struct.pack("i", hparams["n_audio_head"]))
    fout.write(struct.pack("i", hparams["n_audio_layer"]))
    fout.write(struct.pack("i", hparams["n_text_ctx"]))
    fout.write(struct.pack("i", hparams["n_text_state"]))
    fout.write(struct.pack("i", hparams["n_text_head"]))
    fout.write(struct.pack("i", hparams["n_text_layer"]))
    fout.write(struct.pack("i", hparams["n_mels"]))
    fout.write(struct.pack("i", use_f16))

    # write mel filters
    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            fout.write(struct.pack("f", filters[i][j]))

    # write tokenizer
    fout.write(struct.pack("i", len(tokens)))

    for key in tokens:
        fout.write(struct.pack("i", len(key)))
        fout.write(key)

    for name in list_vars.keys():
        data = list_vars[name].squeeze().numpy()
        print("Processing variable: ", name, " with shape: ", data.shape)

        # reshape conv bias from [n] to [n, 1]
        if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
            data = data.reshape(data.shape[0], 1)
            print(f"  Reshaped variable: {name} to shape: ", data.shape)

        n_dims = len(data.shape)

        # looks like the whisper models are in f16 by default
        # so we need to convert the small tensors to f32 until we fully support f16 in ggml
        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype = 1
        if use_f16:
            if n_dims < 2 or \
                    name == "encoder.conv1.bias" or \
                    name == "encoder.conv2.bias" or \
                    name == "encoder.positional_embedding" or \
                    name == "decoder.positional_embedding":
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype = 0
        else:
            data = data.astype(np.float32)
            ftype = 0

        #if name.startswith("encoder"):
        #    if name.endswith("mlp.0.weight") or \
        #       name.endswith("mlp.2.weight"):
        #        print("  Transposing")
        #        data = data.transpose()

        # header
        str_ = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str_), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str_)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: ", fname_out)
    print("")


def pt_to_st(fname_inp, fname_out):
    # try to load PyTorch binary data
    try:
        model_bytes = open(fname_inp, "rb").read()
        with io.BytesIO(model_bytes) as fp:
            checkpoint = torch.load(fp, map_location="cpu")
    except Exception:
        print("Error: failed to load PyTorch model file:", fname_inp)
        sys.exit(1)

    safetensors.torch.save_model(checkpoint, fname_out)


def load_model():
    transcribe = pipeline(task="automatic-speech-recognition", model="BUT-FIT/whisper-ATC-czech-full", chunk_length_s=30, device=device)
    transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(task="transcribe", language="english")

    return transcribe.model


def st_to_pt(fname_inp, fname_out):
    model = load_model()

    state_dict = safetensors.torch.load_file(fname_inp)
    model.load_state_dict(state_dict)

    torch.save(model.state_dict(), fname_out)
    print(f"Model successfully converted from '{fname_inp}' to '{fname_out}'.")


def bin_to_pt(fname_inp, fname_out):
    model = load_model()

    state_dict = torch.load(fname_inp)
    model.load_state_dict(state_dict)

    torch.save(model.state_dict(), fname_out)
    print(f"Model successfully converted from '{fname_inp}' to '{fname_out}'.")


def pretrained_to_pt(fname_out):
    def hf_to_whisper_states(text): # renaming some of the layers so it would work
        text = re.sub('.layers.', '.blocks.', text)
        text = re.sub('.self_attn.', '.attn.', text)
        text = re.sub('.q_proj.', '.query.', text)
        text = re.sub('.k_proj.', '.key.', text)
        text = re.sub('.v_proj.', '.value.', text)
        text = re.sub('.out_proj.', '.out.', text)
        text = re.sub('.fc1.', '.mlp.0.', text)
        text = re.sub('.fc2.', '.mlp.2.', text)
        text = re.sub('.fc3.', '.mlp.3.', text)
        text = re.sub('.fc3.', '.mlp.3.', text)
        text = re.sub('.encoder_attn.', '.cross_attn.', text)
        text = re.sub('.cross_attn.ln.', '.cross_attn_ln.', text)
        text = re.sub('.embed_positions.weight', '.positional_embedding', text)
        text = re.sub('.embed_tokens.', '.token_embedding.', text)
        text = re.sub('model.', '', text)
        text = re.sub('attn.layer_norm.', 'attn_ln.', text)
        text = re.sub('.final_layer_norm.', '.mlp_ln.', text)
        text = re.sub('encoder.layer_norm.', 'encoder.ln_post.', text)
        text = re.sub('decoder.layer_norm.', 'decoder.ln.', text)
        text = re.sub('proj_out.weight', 'decoder.token_embedding.weight', text)
        return text

    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained("BUT-FIT/whisper-ATC-czech-full")
    state_dict = model.state_dict()
    model_dict = {}

    # Rename layers
    for key in list(state_dict.keys())[:]:
        new_key = hf_to_whisper_states(key)
        state_dict[new_key] = state_dict.pop(key)

    # sanity check
    whisper_model = whisper.load_model("medium")
    whisper_model.load_state_dict(state_dict)

    model_dict["dims"] = whisper_model.dims.__dict__ # add mel_spectogram setup to state_dict
    model_dict["model_state_dict"] = state_dict

    print("Converted to whisper")

    torch.save(model_dict, fname_out)
    print("Model successfully saved")


def bin_to_ggml(dir_model, dir_whisper, fname_out):
    conv_map = {
            'self_attn.k_proj'              : 'attn.key',
            'self_attn.q_proj'              : 'attn.query',
            'self_attn.v_proj'              : 'attn.value',
            'self_attn.out_proj'            : 'attn.out',
            'self_attn_layer_norm'          : 'attn_ln',
            'encoder_attn.q_proj'           : 'cross_attn.query',
            'encoder_attn.v_proj'           : 'cross_attn.value',
            'encoder_attn.out_proj'         : 'cross_attn.out',
            'encoder_attn_layer_norm'       : 'cross_attn_ln',
            'fc1'                           : 'mlp.0',
            'fc2'                           : 'mlp.2',
            'final_layer_norm'              : 'mlp_ln',
            'encoder.layer_norm.bias'       : 'encoder.ln_post.bias',
            'encoder.layer_norm.weight'     : 'encoder.ln_post.weight',
            'encoder.embed_positions.weight': 'encoder.positional_embedding',
            'decoder.layer_norm.bias'       : 'decoder.ln.bias',
            'decoder.layer_norm.weight'     : 'decoder.ln.weight',
            'decoder.embed_positions.weight': 'decoder.positional_embedding',
            'decoder.embed_tokens.weight'   : 'decoder.token_embedding.weight',
            'proj_out.weight'               : 'decoder.proj.weight',
    }

    # ref: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    def bytes_to_unicode():
        """
        Returns list of utf-8 byte and a corresponding list of unicode strings.
        The reversible bpe codes work on unicode strings.
        This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
        When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
        This is a significant percentage of your normal, say, 32K bpe vocab.
        To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
        And avoids mapping to whitespace/control characters the bpe code barfs on.
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    dir_model   = Path(dir_model)
    dir_whisper = Path(dir_whisper)
    fname_out   = Path(fname_out)

    encoder = json.load((dir_model / "vocab.json").open("r", encoding="utf8"))
    encoder_added = json.load((dir_model / "added_tokens.json").open("r", encoding="utf8"))
    hparams = json.load((dir_model / "config.json").open("r", encoding="utf8"))

    # Add this block to handle missing 'max_length'
    if "max_length" not in hparams:
        hparams["max_length"] = hparams.get("max_target_positions", 448)

    model = WhisperForConditionalGeneration.from_pretrained(dir_model)

    #code.interact(local=locals())

    n_mels = hparams["num_mel_bins"]
    with np.load(os.path.join(dir_whisper, "whisper/assets", "mel_filters.npz")) as f:
        filters = torch.from_numpy(f[f"mel_{n_mels}"])

    dir_tokenizer = dir_model
    tokens = json.load(open(dir_tokenizer / "vocab.json", "r", encoding="utf8"))

    # use 16-bit or 32-bit floats
    use_f16 = True
    if len(sys.argv) > 4:
        use_f16 = False

    fout = open(fname_out, "wb")

    fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
    fout.write(struct.pack("i", hparams["vocab_size"]))
    fout.write(struct.pack("i", hparams["max_source_positions"]))
    fout.write(struct.pack("i", hparams["d_model"]))
    fout.write(struct.pack("i", hparams["encoder_attention_heads"]))
    fout.write(struct.pack("i", hparams["encoder_layers"]))
    fout.write(struct.pack("i", hparams["max_length"]))
    fout.write(struct.pack("i", hparams["d_model"]))
    fout.write(struct.pack("i", hparams["decoder_attention_heads"]))
    fout.write(struct.pack("i", hparams["decoder_layers"]))
    fout.write(struct.pack("i", hparams["num_mel_bins"]))
    fout.write(struct.pack("i", use_f16))

    fout.write(struct.pack("i", filters.shape[0]))
    fout.write(struct.pack("i", filters.shape[1]))
    for i in range(filters.shape[0]):
        for j in range(filters.shape[1]):
            fout.write(struct.pack("f", filters[i][j]))

    byte_encoder = bytes_to_unicode()
    byte_decoder = {v:k for k, v in byte_encoder.items()}

    fout.write(struct.pack("i", len(tokens)))

    tokens = sorted(tokens.items(), key=lambda x: x[1])
    for key in tokens:
        text = bytearray([byte_decoder[c] for c in key[0]])
        fout.write(struct.pack("i", len(text)))
        fout.write(text)

    list_vars = model.state_dict()
    for name in list_vars.keys():
        # this seems to not be used
        # ref: https://github.com/huggingface/transformers/blob/9a5b84a0076a04fe9596da72e8668069d4f09ea0/src/transformers/models/whisper/modeling_whisper.py#L1099-L1106
        if name == "proj_out.weight":
            print('Skipping', name)
            continue

        src = name

        nn = name
        if name != "proj_out.weight":
            nn = nn.split(".")[1:]
        else:
            nn = nn.split(".")

        if nn[1] == "layers":
            nn[1] = "blocks"
            if ".".join(nn[3:-1]) == "encoder_attn.k_proj":
                mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
            else:
                mapped = conv_map[".".join(nn[3:-1])]
            name = ".".join(nn[:3] + [mapped] + nn[-1:])
        else:
            name = ".".join(nn)
            name = conv_map[name] if name in conv_map else name

        print(src, ' -> ', name)
        data = list_vars[src].squeeze().numpy()
        data = data.astype(np.float16)

        # reshape conv bias from [n] to [n, 1]
        if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
            data = data.reshape(data.shape[0], 1)
            print("  Reshaped variable: ", name, " to shape: ", data.shape)

        n_dims = len(data.shape)
        print(name, n_dims, data.shape)

        # looks like the whisper models are in f16 by default
        # so we need to convert the small tensors to f32 until we fully support f16 in ggml
        # ftype == 0 -> float32, ftype == 1 -> float16
        ftype = 1
        if use_f16:
            if n_dims < 2 or \
                    name == "encoder.conv1.bias" or \
                    name == "encoder.conv2.bias" or \
                    name == "encoder.positional_embedding" or \
                    name == "decoder.positional_embedding":
                print("  Converting to float32")
                data = data.astype(np.float32)
                ftype = 0
        else:
            data = data.astype(np.float32)
            ftype = 0

        # header
        str_ = name.encode('utf-8')
        fout.write(struct.pack("iii", n_dims, len(str_), ftype))
        for i in range(n_dims):
            fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
        fout.write(str_)

        # data
        data.tofile(fout)

    fout.close()

    print("Done. Output file: ", fname_out)
    print("")
