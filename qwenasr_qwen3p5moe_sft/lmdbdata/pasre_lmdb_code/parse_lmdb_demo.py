
import os
import sys


import json
import lmdb
import numpy as np
import random
from protofiles import ASRProto
import re
import wave
from file_io import LengthsFileReader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

#原始protobuf为6.33.6，在这里降级为了3.20.3

def is_bytes(obj):
    return isinstance(obj, bytes)


class GemmaTokenizer(object):
    def __init__(self, vocab_file: str):
        assert os.path.exists(vocab_file), \
            f"vocab file path ({vocab_file}) is not exist"
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file)
        self.pad_id = self.tokenizer.pad_token_id
        self.unk_id = 3
        self.sep_id = 2 #self.encoder['<s>'] # 1       sep_id
        self.eod_id = 1 #self.encoder['</s>'] # 2       eod_id

    def add_space(self, text):
        text=re.sub("(，|。|！|？).*",r"\1 ",text)
        return text

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def __len__(self):
        return len(self.tokenizer)

    @property
    def unk(self):
        return self.unk_id

    @property
    def eod(self):
        return self.eod_id

    def tokenize(self, text):
        return self.tokenizer.encode(text)
    
    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.decode(ids)

    def detokenize(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens, skip_special_tokens=True):
        if len(tokens) == 0:
            text = self.tokenizer.decode(tokens, skip_special_tokens)
        else:
            if isinstance(tokens[0], list):
                text = []
                for subtoken in tokens:
                    text.append(self.tokenizer.decode(subtoken, skip_special_tokens))
            else:
                text = self.tokenizer.decode(tokens, skip_special_tokens)
        return text


tokenizer = GemmaTokenizer('/data/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/lmdbdata/res')


filein=open('/data/yjjiang11/work/vibecoding/Qwen3.5/qwenasr_qwen3p5moe_sft_lmdb/lmdbdata/train.json')
random.seed(12345)

jconts=json.load(filein)

for jcont in jconts['datasets']:

    wavjson = jcont['wav']
    filewav=open(wavjson)
    curjconts = json.load(filewav)
    input_wav_mdb = curjconts['lmdb_path']
    input_len_bin = curjconts['metas']['lengths_file']
    lengths=[]
    length_reader = LengthsFileReader(input_len_bin).open()
    for n in length_reader:
        lengths.append(n)

    # curitems = input_wav_mdb.split('/')
    curitems = wavjson.split('/')
    curdir = '@@'.join(curitems[-5:-1])+'@@'+curitems[-1].split('.')[0]
    if 'range' in jcont:
        curdir+='_'+str(jcont['range'][0])+'-'+str(jcont['range'][1])

    wav_env = lmdb.open(input_wav_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    wav_txn = wav_env.begin()
    keydict = {}
    num=0
    idx=0
    os.system('mkdir -p wavs/'+curdir)
    os.system('mkdir -p lengths')

    filelen = open('lengths/'+curdir+'.txt','w')
    for key, value in wav_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        mlf_proto = ASRProto.FromString(value)
        # wav
        if mlf_proto.data_type==5:
            filelen.write(key+'\t'+str(lengths[idx])+'\n')
            mlf_byte_wav = mlf_proto.data
            f00=wave.open('./wavs/'+curdir+'/'+key+'.wav','wb')
            f00.setnchannels(1)
            f00.setsampwidth(2)
            f00.setframerate(16000)
            f00.writeframes(mlf_byte_wav)
            f00.close()
            num+=1
            keydict[key]=1
            if num >20:
                break
        idx += 1
    wav_env.close()



    edjson = jcont['ed_label']
    os.system('mkdir -p ed_label')
    fileed=open(edjson)
    curjconts = json.load(fileed)
    input_ed_mdb = curjconts['lmdb_path']
    ed_env = lmdb.open(input_ed_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    ed_txn = ed_env.begin()
    fileout=open('ed_label/'+curdir+'.txt','w')
    for key, value in ed_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        if key in keydict:
            mlf_proto = ASRProto.FromString(value)
            mlf_byte = mlf_proto.data
            mlf_logit = np.frombuffer(mlf_byte, dtype=np.int32)
            mlf_logit = mlf_logit
            mlf_logit = mlf_logit.tolist()

            word = tokenizer.decode(mlf_logit)
            fileout.write(key+'\t'+word+'\n')
                
    ed_env.close()



    edjson = jcont['ed_label_src']
    os.system('mkdir -p ed_label_src')
    fileed=open(edjson)
    curjconts = json.load(fileed)
    input_ed_mdb = curjconts['lmdb_path']
    ed_env = lmdb.open(input_ed_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    ed_txn = ed_env.begin()
    fileout=open('ed_label_src/'+curdir+'.txt','w')
    for key, value in ed_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        if key in keydict:
            mlf_proto = ASRProto.FromString(value)
            mlf_byte = mlf_proto.data
            mlf_logit = np.frombuffer(mlf_byte, dtype=np.int32)
            mlf_logit = mlf_logit
            mlf_logit = mlf_logit.tolist()
            word = tokenizer.decode(mlf_logit)
            fileout.write(key+'\t'+word+'\n')
    ed_env.close()


    edjson = jcont['ce_label']
    os.system('mkdir -p ce_label')
    fileed=open(edjson)
    curjconts = json.load(fileed)
    input_ed_mdb = curjconts['lmdb_path']
    ed_env = lmdb.open(input_ed_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    ed_txn = ed_env.begin()
    fileout=open('ce_label/'+curdir+'.txt','w')
    for key, value in ed_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        if key in keydict:
            mlf_proto = ASRProto.FromString(value)
            mlf_byte = mlf_proto.data
            mlf_logit = np.frombuffer(mlf_byte, dtype=np.int32)
            mlf_logit = mlf_logit
            mlf_logit = mlf_logit.tolist()
            word = tokenizer.decode(mlf_logit)
            fileout.write(key+'\t'+word+'\n')
    ed_env.close()


    edjson = jcont['ed_label_kws']
    os.system('mkdir -p ed_label_kws')
    fileed=open(edjson)
    curjconts = json.load(fileed)
    input_ed_mdb = curjconts['lmdb_path']
    ed_env = lmdb.open(input_ed_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    ed_txn = ed_env.begin()
    fileout=open('ed_label_kws/'+curdir+'.txt','w')
    for key, value in ed_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        if key in keydict:
            mlf_proto = ASRProto.FromString(value)
            mlf_byte = mlf_proto.data
            mlf_logit = np.frombuffer(mlf_byte, dtype=np.int32)
            mlf_logit = mlf_logit
            # import pdb;pdb.set_trace()
            mlf_logit = mlf_logit.tolist()
            word = tokenizer.decode(mlf_logit)
            fileout.write(key+'\t'+word+'\n')
    ed_env.close()


    edjson = jcont['ed_label_asr']
    os.system('mkdir -p ed_label_asr')
    fileed=open(edjson)
    curjconts = json.load(fileed)
    input_ed_mdb = curjconts['lmdb_path']
    ed_env = lmdb.open(input_ed_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    ed_txn = ed_env.begin()
    fileout=open('ed_label_asr/'+curdir+'.txt','w')
    for key, value in ed_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        if key in keydict:
            mlf_proto = ASRProto.FromString(value)
            mlf_byte = mlf_proto.data
            mlf_logit = np.frombuffer(mlf_byte, dtype=np.int32)
            mlf_logit = mlf_logit
            mlf_logit = mlf_logit.tolist()
            word = tokenizer.decode(mlf_logit)
            fileout.write(key+'\t'+word+'\n')
    ed_env.close()


    edjson = jcont['ed_label_acc']
    os.system('mkdir -p ed_label_acc')
    fileed=open(edjson)
    curjconts = json.load(fileed)
    input_ed_mdb = curjconts['lmdb_path']
    ed_env = lmdb.open(input_ed_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    ed_txn = ed_env.begin()
    fileout=open('ed_label_acc/'+curdir+'.txt','w')
    for key, value in ed_txn.cursor():
        if is_bytes(key):
            key=key.decode()
        if key in keydict:
            mlf_proto = ASRProto.FromString(value)
            mlf_byte = mlf_proto.data
            mlf_logit = np.frombuffer(mlf_byte, dtype=np.int32)
            mlf_logit = mlf_logit
            fileout.write(key+'\t'+str(mlf_logit)+'\n')
    ed_env.close()