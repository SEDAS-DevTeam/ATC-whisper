
# ATC-whisper

ATC port for robust whisper model

## Project structure

``` python
src
├─ dataset
├─ infer
│   └─ main.cpp # main file for whisper inference
├─ model
├─ project.py # simple project manager
└─ train
    ├─ data.py # for dataset preparation
    ├─ main.py # main file for whisper training
    └─ model.py # for whisper model initialization
```

## Installation/Usage

Installation steps are located in the [SEDAS documentation](https://sedas-docs.readthedocs.io/en/latest/user-manual.html#app-installation)

## Dataset modifications

### Excluded annotations from ATCOSIM

(done automatically using script)

- `<FL></FL>`
    - when empty (full french language)
    - when inserted into another sequence using <FL> tags
- `[empty]` - if empty completely

### Cleaned annotations from text

- `[HNOISE]` - irrelevant
- `*=` and `=*` - breaks in words are going to be joined together
- `@*` - @ removed from words
- `[EMPTY]` - replaced with empty text file
- `[FRAGMENT]` - irrelevant
- `[NONSENSE]` - irrelevant
- `[UNKNOWN]` - irrelevant
- `~*` - Ignore (Currently, we will only use NATO Alphabet + some acronyms)
- `<OT></OT>` - removed completely from annotations

## TODO

- [x] Add Whisper download
- [x] Add Dataset processing
- [x] Dataset cleanup
- [x] Finish infer
- [x] Add model stashing
- [x] Finish whisper loading to torch (preparing for fine-tune)
- [x] Add wav file parsing and validate
- [x] Cleanup TODOs
- [x] Recheck everything + add DataCollator
- [x] Fix the conversion, rework pt-to-ggml.py
- [x] If conversion works maybe move download_whisper_repo to download_model?
- [x] Find a way of downloading larger models
- [x] Add whisper.cpp and openai/whisper as submodules
- [ ] Finish C++ inference testing script
- [ ] Implement workspace switch on testing and also on training
- [ ] Switch from conda to pyenv
- [x] Add Huggingface publishing
- [ ] Add progressbar on huggingface upload
