"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
current_dir = os.getcwd()
input_file_path = os.path.join(current_dir, 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/js-ts/nanoGPT/master/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters: 13,352,211
# all the unique characters: 
#  !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz|}~¡¢£¥§©°±²³´µ·º½¾¿ÁÄÅÆÇÉÍÎÑÓÖ×ØÚÜÞßàáâãäåæçèéêëìíîïðñòóôöøùúûüýþĀāăąćČčĐđĒēĔĕėęěğħīİıĽŁłńŌōœřŚśŞşŠšţũūůųźŻżŽžơưǎǐǔǜȘșțȯɐɑɒɔəɛɜɡɢɪɫɳɽɾʁʃʊʋʒʔʕʲʻʼʾʿˈˌː̥̯͍́̃̍͘ΑΔΚΝΠΤΦΧΩάέήίαβγδεηθικλμνξοπρςστυχψωόύώАБВГКПРСУХЯавгдежзийклмнопрстухцшъыьюяєֵֶָֹּאבגהוזחילםמןנסףפצרשתءأإابةتثجحخدذرسشصعفقكلمنهويڠܐܕܗܝܠܢܬܲܵंअईकगणतदनपबमयरलवषसहािुूेैो्আলহা্ਅਲਹਾੁੱഅളഹാ്กคงชซญฐณดตนบปพภมยรลวศษสหอฮะัาิีึเแไ็่้๊์ဂစဇတနပမရလအာို္်ြွ၁၂၇၈ႠႢႣႨႬႵႿაბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰჱჲჳჴჵჶჷჸჹჺ჻ᵻḍḏḤḥḷṃṅṇṛṣṭṯạảấầẩậắễệịọỏỗộớứửữỳỹἀἰὀὁὑ​‐‑–—―‘’“”„†‡•…′″⁄₣₤₩€₱₹⅓⅔→−≡≤①☉☫♀♭♯⚳〈〉〜あいうおかがきぎくぐけこごさしすずぜただちっつとなにのはばひふほまみめもゃゆょらりるれわをんァアィイゥェエォオカガキクグゲコサシジスズセタダチッツテデトドナニネノハバパフブプヘベマミムモャュョラリルレロンヴ・ー一七下世丙中为主乃之乙九二云人今付伊似作侗依信傳儚充光全八六兵其具円再出判利制刷前剛劇劉助動勢包化北十千华南印厂去古可台史同名君吳周命和咲唐善四國園圣在坂型堂堤場塘士夕大天夫奈套女妙姚子孟学學守安宋完宗定宝宫寝寶寺小少尾山岳川州巳市師平广庆府座廬建开式張彌彩影彼征律後御微德心必忠思愛憑憶應懷成战戦所扈技拉拱拳挑揺攻放政散文斯方日旦旭昌明星春晋景智曦書月望朝未本李村杜束来東板林果桜梶棘椎楊楚榮樸橘機正殻殿毅母民水汉沂沙河法泗波泣洪浮淹清湯漢澄澤火灯灵灼無焼熱牌物狐狸玄玉王玩珂珙球理琦琪瓊生甫田甲畢番畫疆病瘡白皇皮真砲礮祈神祠秋秘稽空立竹箋籠精紀約統絵緬織繹義翠者耕聖肖背胡膀臂致興舍艦良芥花芳芽若英萬著藕藥蘄蘇蘿處蝴蝶行術衛表裁裝要規覺观解言記詔詩誓誡誰謎譌譚譜變许谭豪豫費贈贵赤趙足跡軒転辛辨迪逆遇運過道遠選邦邱邵部郭都酈里野金鉄銃鋼錄錡錦鍵鐵钱铁關防阿陈陳陽隊階集雞雪雲霖霹靂韓頒願類顯颜饾駢驗马體高魯齋龍대독라립병보부알연제ﷲﻋ／３～�
# vocab size: 1,153
# train has 12,016,989 tokens
# val has 1,335,222 tokens