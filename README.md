# SER_GE2E

## Preprocess databases (IEMOCAP, EMODB, SHEMO)
1. download the dataset
2. create the folder named "train_audio_<eng/ger/per>/<IEMOCAP/EMODB/SHEMO>"
3. run the Data_Processing.ipynb to reorganize the datasets to the following structure:
```
   - train_audio_<eng/ger/per> 
      - <IEMOCAP/EMODB/SHEMO>
          - emotion A
              - wav       
                  - audio A_1        
                  - audio A_2            
          - emotion B
              - wav       
                  - audio B_1         
                  - audio B_2
          ...
              
```
4. run the command
```
python encoder_preprocess.py ./train_audio_<eng/ger/per>
```
5. The output folder is named SV2TTS, which can also be found in the drive:
    - IEMOCAP SV2TTS: https://drive.google.com/file/d/1AecA1KGuIW8t0czRmzvxI0gC7zsZ9joe/view?usp=sharing
    - EMODB SV2TTS: https://drive.google.com/file/d/1NEQx8T5jgRKbKGYZRgKKT9eygS1MrH54/view?usp=sharing
    - SHEMO SV2TTS: https://drive.google.com/file/d/1MNt_JEFmpOqG9xWgjUrrDF7q9095b5pE/view?usp=sharing


Note: In order to lost the data during processing, I changed the `partials_n_frames` in `encoder/params_data.py`. Currently don't know what would be affected.


## Train GE2E model
1. will need to download the `visdom` package to visualize the training process. And start the visdom server before running the command.
2. change the parameter `speakers_per_batch` equals to the number of emotion classes in `encoder/params_model.py`. `utterances_per_speaker` can be tuned.
3. run the command:
```
python encoder_train.py <my_run> <datasets_root>/SV2TTS/encoder
```

`<my_run>` is like a session name, can be defined as any name. And the training process can be seen in the visdom port.

`visdom` can be disabled using `--no_visdom`


## Generate GE2E embedding
There is a pretrained model in the `saved_models/default` folder if don't want to train from scratch. 

run the command: 
```
python generate_embed.py
```
Can use the following options:
  - `-s`: the path to the source audio (e.g. `./train_audio_eng/IEMOCAP`)
  - `-d`: the path to the destination embedding.

