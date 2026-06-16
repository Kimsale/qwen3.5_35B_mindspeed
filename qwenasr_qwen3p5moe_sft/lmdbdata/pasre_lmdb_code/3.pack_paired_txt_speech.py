import sentencepiece as spm
import lmdb
import json
import numpy as np
import os
import wave
import soxr
import sys
import time
# sys.path.insert(0, '/wx-mix01/mtoast/permanent/suiliang/data_pad_for_yf')
from file_io import LengthsFileReader
from protofiles import ASRProto
import random
import re
import pdb
from tqdm import tqdm
from transformers import AutoTokenizer
import string
import shutil
from lengthsfilewriter import lengths_bin_generator
from file_io import LengthsFileReader
import ast

class Tokenizer(object):
    """
    Tokenizing and encoding/decoding text using the hugging face type tokenizer.
    """
    def __init__(self, vocab_file: str):
        assert os.path.exists(vocab_file), \
                f"vocab file path ({vocab_file}) is not exist"

        added_token  = []
        added_token += [f"<unused{i}>" for i in range(20, 50)]

        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, additional_special_tokens=added_token, use_fast=False)

        self.pad_id = 151643
        self.unk_id = 128244

        # self.eos_id: int = self.tokenizer.eos_token_id
        # self.eod_id = self.eos_id

        self.sep_id = 151644 #self.encoder['<s>']  # 1       sep_id
        self.eod_id = 151645 #self.encoder['</s>']  #2       eod_id
        # self.pad_id = self.encoder['<pad>']  # 0
        # self.unk_id = self.encoder["<unk>"] #3

    def add_space(self, text):
        text=re.sub("(，|。|！|？) *",r"\1 ",text)
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

    def decode(self, tokens, skip_special_tokens=False):
        text = self.tokenizer.decode(tokens, skip_special_tokens)
        return text


tokenizer_path = "/wx-mix01/mtoast/permanent/suiliang/si-llm/resources/qwen-omni/"
qwen_tokenizer = Tokenizer(tokenizer_path)

txt_path = "/wx-mix01/mtoast/permanent/suiliang/data_pad_for_yf_moshi/yufan_acou_en2cn_feiguding_speech_txt_token.json"
with open(txt_path, 'r') as mdb_p:
    process_names = json.load(mdb_p)

out_json_list = []
iii = 0
for in_wav_mdb in tqdm(process_names):
    # iii += 1
    # if iii > 1:
    #     break
    out_wav_mdb = in_wav_mdb.copy()  # 使用copy避免修改原始数据
    if "ce_label" in out_wav_mdb:
        del out_wav_mdb["ce_label"]

    wav_bin_path = in_wav_mdb["wav"]
    with open(wav_bin_path, "r") as f_wav_bin:
        bin_json = json.load(f_wav_bin)
    bin_path_in = bin_json["metas"]["lengths_file"]
    in_wav_mdb_path = bin_json["lmdb_path"]

    trans_path = in_wav_mdb["ed_label"]
    with open(trans_path, "r") as f_trans:
        in_mlf_rlt_mdb = json.load(f_trans)["lmdb_path"]

    ratio_path = in_wav_mdb["ed_label_ratio"]
    with open(ratio_path, "r") as f_ratio:
        in_mlf_ratio_mdb = json.load(f_ratio)["lmdb_path"]

    in_mdb_dir = "/".join(trans_path.split("/")[:-2])
    mdb_id = trans_path.split("/")[-1].split('.')[0]
    file_name = trans_path.replace("/wx-mix01/mtoast/permanent/suiliang/data_lc/from_hwy/data_with_asr/", "").replace("/wx-mix01/mtoast/permanent/suiliang/data_lc/yufan_new/", "yf/all/").replace("/","@").replace(".json","")

    # if ( "chinese" in trans_path or "cn2en" in trans_path ) or "22kh_wav_cmn_fa_rmerrsent" in trans_path: #中到英
    mdb_in_type = "chinese"   
    mdb_out_type = "english"   
    # elif  ("english" in trans_path or "en2cn" in trans_path or "muldomain" in trans_path):      # 英到中
    # mdb_in_type = "english"
    # mdb_out_type = "chinese"
    # else:
    #     print("mdb_out_type found error")
    #     sys.exit()

    if mdb_out_type == "chinese":
        output_mdb_dir = "/wx-mix01/mtoast/permanent/suiliang/data_pad_for_yf_moshi/mdb_pertxttoken_speechtoken/chinese/" + file_name.replace("@", "/")
        speech_txt_token_path = f"/wx-mix01/mtoast/permanent/suiliang/data_pad_for_yf/results_pertxttoken_speechtoken_merge_mark_nomatch/chinese/{file_name}.txt"
    elif mdb_out_type == "english":
        output_mdb_dir = "/wx-mix01/mtoast/permanent/suiliang/data_pad_for_yf_moshi/mdb_pertxttoken_speechtoken/english/" + file_name.replace("@", "/")
        speech_txt_token_path = f"/wx-mix01/mtoast/permanent/suiliang/data_pad_for_yf/results_pertxttoken_speechtoken_merge_mark_nomatch/english/{file_name}.txt"

    print("Processing: ", in_mdb_dir, " || ", output_mdb_dir)
    os.makedirs(output_mdb_dir+'/lmdb', exist_ok=True)

    output_trans_rlt_mdb =  output_mdb_dir +  "/lmdb/" +  mdb_id + ".mlf_trans_rlt_for_gemma3.mdb" 
    output_trans_rlt_mdb_json =  output_mdb_dir +  "/lmdb/" +  mdb_id + ".mlf_trans_rlt_for_gemma3.mdb.json"
    output_trans_ratio_mdb =  output_mdb_dir +  "/lmdb/" +  mdb_id + ".mlf_trans_ratio_for_cs3.mdb" 
    output_trans_ratio_mdb_json =  output_mdb_dir +  "/lmdb/" +  mdb_id + ".mlf_trans_ratio_for_cs3.mdb.json"

    if os.path.exists(output_trans_rlt_mdb_json):
        print(output_trans_rlt_mdb_json, "------>has exist")
        out_wav_mdb["ed_label"] = output_trans_rlt_mdb_json
        out_wav_mdb["ed_label_ratio"] = output_trans_ratio_mdb_json
        with open(output_trans_rlt_mdb_json, "r") as f_num:
            data_num = json.load(f_num)["total_num"]
        out_wav_mdb["data_num"] = data_num
        out_json_list.append(out_wav_mdb)
        continue
    if os.path.exists(output_trans_rlt_mdb):
        # os.remove(process_name_out_dir)
        shutil.rmtree(output_mdb_dir)
        os.makedirs(output_mdb_dir+'/lmdb',exist_ok=True)

    mlf_rlt_content_dict = {}
    try:  
        mlf_rlt_lmdb_env = lmdb.open(in_mlf_rlt_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    except: 
        print('------- have no mlf_rlt')
        exit()
    mlf_rlt_lmdb_txn = mlf_rlt_lmdb_env.begin()
    for key, value in mlf_rlt_lmdb_txn.cursor():
        proto = ASRProto.FromString(value)
        mlf_rlt_content_dict[key] = {}
        mlf_rlt_content_dict[key]["name"] = proto.name
        mlf_rlt_content_dict[key]["data"] = proto.data
    mlf_rlt_lmdb_env.close()
    ori_key_list_rlt = set( mlf_rlt_content_dict.keys() )
    print("mlf_rlt mdb read end")
    del mlf_rlt_lmdb_txn

    mlf_ratio_content_dict = {}
    try:  
        mlf_ratio_lmdb_env = lmdb.open(in_mlf_ratio_mdb, map_size=1099511627776, max_dbs=2, subdir=False, readonly=True, lock=False)
    except: 
        print('------- have no mlf_ratio')
        exit()
    mlf_ratio_lmdb_txn = mlf_ratio_lmdb_env.begin()
    for key, value in mlf_ratio_lmdb_txn.cursor():
        proto = ASRProto.FromString(value)
        mlf_ratio_content_dict[key] = {}
        mlf_ratio_content_dict[key]["name"] = proto.name
        mlf_ratio_content_dict[key]["data"] = proto.data
    mlf_ratio_lmdb_env.close()
    ori_key_list_ratio = set( mlf_ratio_content_dict.keys() )
    print("mlf_ratio mdb read end")
    del mlf_ratio_lmdb_txn

    mlf_trans_rlt_env = lmdb.open(output_trans_rlt_mdb, map_size=1099511627776, max_dbs=2, subdir=False)
    mlf_trans_rlt_txn = mlf_trans_rlt_env.begin(write=True)
    mlf_trans_ratio_env = lmdb.open(output_trans_ratio_mdb, map_size=1099511627776, max_dbs=2, subdir=False)
    mlf_trans_ratio_txn = mlf_trans_ratio_env.begin(write=True)
    
    cnt = 0
    total_num = 0
    num_skip_rlt = 0
    bin_lengths_new = []
    record_trans_token = np.array([])
    count = 0
    drop_num = 0
    for key in mlf_rlt_content_dict:

        name = mlf_rlt_content_dict[key]["name"]
        key_new = "{:011d}".format(cnt).encode()

        trans_token = mlf_rlt_content_dict[key]["data"]
        trans_token = np.frombuffer(trans_token, dtype=np.int32)

        ratio_token = mlf_ratio_content_dict[key]["data"]
        ratio_token = np.frombuffer(ratio_token, dtype=np.int32)

        # TODO 在这里开始进行替换相关逻辑
        # 根据step.md的处理要求：
        # 1. 去掉开头结尾的PAD
        # 2. 剩余数据中，去掉最大ratio大于20的数据
        # 3. 将所有的<PAD>替换为<sil>
        # 4. 对trans token执行ratio调整：如果ratio大于5，则插入<PAD>平分ratio

        PAD_ID = 151671  # <PAD> token id <tts_text_eod>
        SIL_ID = 151675   # <sil> token id <tts_pad>
        SIL_EOD_ID = 151673

        # Step 1: 去掉开头结尾的PAD
        # 找开头连续PAD的数量
        start_pad_count = 0
        for t in trans_token:
            if t == 151673:
                start_pad_count += 1
            else:
                break

        # 找结尾连续PAD的数量
        end_pad_count = 0
        for t in reversed(trans_token):
            if t == 151673:
                end_pad_count += 1
            else:
                break

        # 去掉开头和结尾的PAD
        if start_pad_count > 0 or end_pad_count > 0:
            trans_token = trans_token[start_pad_count:len(trans_token)-end_pad_count if end_pad_count > 0 else len(trans_token)]
            ratio_token = ratio_token[start_pad_count:len(ratio_token)-end_pad_count if end_pad_count > 0 else len(ratio_token)]

        # Step 2: 去掉最大ratio大于20的数据
        if len(ratio_token) == 0 or ratio_token.max() > 20:
            drop_num += 1
            # trans_token = np.array([], dtype=np.int32)
            # ratio_token = np.array(new_ratio_token, dtype=np.int32)
            # continue
        else:
            # Step 3: 将所有的<PAD>替换为<sil>
            trans_token = np.where(trans_token == 151673, SIL_ID, trans_token)

            # Step 4: 对trans token执行ratio调整
            # 如果某个trans token的ratio小于等于5，则不做替换
            # 如果大于5，则该token后插入<PAD>平分该字的ratio，使得该字的ratio小于等于5
            # 例如"我"这个字的ratio是17，则需要插入3个<PAD>，"我"分2，三个<PAD>平分后面的15，变为2、5、5、5
            new_trans_token = []
            new_ratio_token = []

            for i in range(len(trans_token)):
                t_token = trans_token[i]
                r_ratio = ratio_token[i]

                if r_ratio <= 5:
                    # ratio <= 5，不做替换
                    new_trans_token.append(t_token)
                    new_ratio_token.append(r_ratio)
                else:
                    # ratio > 5，需要插入PAD
                    # 当前字保留 ratio % 5 (如果余数为0则保留5)
                    remainder = r_ratio % 5
                    if remainder == 0:
                        remainder = 5
                    # 需要插入的PAD数量
                    num_pads = (r_ratio - remainder) // 5

                    # 当前字保留余数
                    new_trans_token.append(t_token)
                    new_ratio_token.append(remainder)

                    # 插入PAD，每个PAD的ratio为5
                    for _ in range(num_pads):
                        if _ == num_pads -1:
                            new_trans_token.append(SIL_EOD_ID)
                            new_ratio_token.append(5)
                        else:
                            new_trans_token.append(PAD_ID)
                            new_ratio_token.append(5)

            trans_token = np.array(new_trans_token, dtype=np.int32)
            ratio_token = np.array(new_ratio_token, dtype=np.int32)

            # print(trans_token)
            # print(ratio_token)
            # exit(0)

        # 开始生产lmbd
        trans_rlt = np.array(trans_token, dtype=np.int32)
        trans_rlt_proto = ASRProto()
        trans_rlt_proto.name = name
        trans_rlt_proto.data_type = ASRProto.INT32
        trans_rlt_proto.dim = 1 
        trans_rlt_proto.data = trans_rlt.tobytes()
        mlf_trans_rlt_txn.put(key_new, trans_rlt_proto.SerializeToString())
        if cnt % 10000 == 0: 
            mlf_trans_rlt_txn.commit()
            mlf_trans_rlt_txn = mlf_trans_rlt_env.begin(write=True)

        trans_ratio = np.array(ratio_token, dtype=np.int32)
        trans_ratio_proto = ASRProto()
        trans_ratio_proto.name = name
        trans_ratio_proto.data_type = ASRProto.INT32
        trans_ratio_proto.dim = 1 
        trans_ratio_proto.data = trans_ratio.tobytes()
        mlf_trans_ratio_txn.put(key_new, trans_ratio_proto.SerializeToString())
        if cnt % 10000 == 0: 
            mlf_trans_ratio_txn.commit()
            mlf_trans_ratio_txn = mlf_trans_ratio_env.begin(write=True)

        cnt += 1
        total_num += 1

    print("cnt: ",cnt,"total_num: ",total_num, "drop: ", drop_num, "ratio: ", round(drop_num/total_num,4))

    total_num = cnt
    if cnt > 0:
        mlf_trans_rlt_txn.commit()
        mlf_trans_rlt_json = {
            "lmdb_path": output_trans_rlt_mdb,
            "total_num": total_num,
            "is_subdir": "false",
            "increment_key": "true",
            "language": mdb_out_type
        }
        with open(output_trans_rlt_mdb_json, 'w') as f:
            json.dump(mlf_trans_rlt_json, f, indent="    ")
        mlf_trans_rlt_env.close()

        mlf_trans_ratio_txn.commit()
        mlf_trans_ratio_json = {
            "lmdb_path": output_trans_ratio_mdb,
            "total_num": total_num,
            "is_subdir": "false",
            "increment_key": "true",
            "language": mdb_out_type
        }
        with open(output_trans_ratio_mdb_json, 'w') as f:
            json.dump(mlf_trans_ratio_json, f, indent="    ")
        mlf_trans_ratio_env.close()

        print("over, total num", total_num)


    out_wav_mdb["ed_label"] = output_trans_rlt_mdb_json
    out_wav_mdb["ed_label_ratio"] = output_trans_ratio_mdb_json

    out_json_list.append(out_wav_mdb)

with open(txt_path.replace(".json", "_txtpad.json"), 'w') as f:
    json.dump(out_json_list, f, indent="    ", ensure_ascii=False)
# exit(0)
