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

## Dataset modifications

### Excluded annotations from ATCOSIM
(done automatically using script)

- `<FL></FL>`
    - when empty (full french language)
    - when inserted into another sequence using <FL> tags
- `<OT></OT>` - only when e

### Cleaned annotations from text
- `[HNOISE]` - irrevelant

## TODO

- [x] Add Whisper download
- [ ] Add Dataset processing
- [ ] Finish infer
- [x] Add model stashing
- [ ] Finish whisper loading to torch (preparing for fine-tune)