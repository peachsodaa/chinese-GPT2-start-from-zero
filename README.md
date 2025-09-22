Training GPT2 Chinese from zero to hero
==

1.Description:
---
AI Design Entrance Exam 从头训练一个82M的中文GPT2模型，使用BERT的Tokenizer.中文语料采用小说《遮天》的内容，大小约19.8MB。训练15个周期，batchsize=8。最终可以续写10句以上的遮天小说。

2.Start:
----
(1)***environment***

首先下载依赖。
```bash
pip install -r requirements.txt
```

(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料由.txt文件更改为input3.json文件

按照参考样例./train3.json更改input3.json文件格式,由于数据集内容为原始的小说内容，包含着大量的非法字符和json读取不支持的控制字符，因此我们对原始数据集文件进行处理，去除其中非法字符，生成预处理好的数据集文件train3.json。
```bash
python clr_ctrl.py
```

(3)***Model***

在model_config 定义初始GPT-2模型的超参数配置，
>- "initializer_range": 0.02 ： 定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为0，标准差为0.02的正态分布中进行随机初始化。
>- "layer_norm_epsilon": 1e-05 ： 用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为1e-05，用于稳定训练。
>- "n_ctx": 1024 ： 表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为1024，即模型一次最多能处理1024个 token。
>- "n_embd": 768 ： 表示每个token的嵌入维度大小，即模型中词向量的维度。设置为768，即每个词汇的表示向量是768维的。
>- "n_head": 12 ： 表示自注意力机制中的注意力头的数量。设置为12，即模型的多头注意力机制中有12个独立的头。
>- "n_layer": 10 ： 表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
>- "n_positions": 1024 ： 表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 n_ctx一致，表示模型最多能处理1024个位置的token。
>- "vocab_size": 13317 ： 表示词汇表的大小，即模型可以识别和生成的词汇数量。在这里，词汇表大小为 21128，表示该模型可以处理的词汇量为21128个不同的 token。


(4)***Training***

现在，使用处理好的数据集来训练我们的初始gpt2模型，使用如下命令：
```bash
python train.py   --model_config config/model_config_small.json   --tokenized_data_path data/tokenized/   --tokenizer_path cache/vocab_small.txt   --raw_data_path data/train3.json   --epochs 15   --log_step 200   --stride 512   --output_dir model/   --device 0,1   --num_pieces 100   --raw
```

训练使用的硬件为NVIDIA Geforce RTX 4070 SUPER，训练用时约30h，在这个过程中，我们可以看到命令窗口打印出模型的config文件，定义了模型的结构；同时也打印出了模型的参数量，为81894144，约82M


```json
Print Model config
config:
{
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 10,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 13317
}
number of parameters: 81894144
```

训练过程中，每个epoch对应的模型都将存储在./model/目录下，最终训练好的模型将存储在./model/final_model/路径中。

(5)***Generate***

现在，我们可以使用我们用目标语料训练生成的模型来进行文字生成，使用如下命令：
```bash
python generate.py   --device 0,1   --length 1000   --tokenizer_path cache/vocab_small.txt   --model_path model/final_model   --prefix "[CLS]叶凡"   --topp 1   --temperature 1.0 --save_samples --save_samples_path ./mnt/
```

3.Result
--
最终会生成10个文字样本，存储在./mnt/目录下，其中之一如下：

======================================== SAMPLE 1 ========================================

叶凡与庞博面面相觑，两人相互看了一眼。“叶子你在做什么？”王子文也很震惊。“不是我的想象，你们应该是在奇士府中的一处山门，我也没有办法告诉你。”李黑水道。“叶子你也太小气了，这里面有大凶险，想要远观察个透彻。”庞博很不满。“我不想多说什么了，你要是没有办法，这里有五色祭坛，你要是能够活下去，我不会让这里仙宫遗迹下不再是遗迹，想想去看你的本皇教训几眼。”王艳笑着说道。“你别这样做，王冲你赶紧离开吧，我不是要去西漠。”黑皇道。“那里，我要去西漠，你去找什么，别的话肯定有一定的地方坐标。”叶凡道。“我去找看一看，你真想进去吗？”李黑水与柳寇等人一起离开。庞博很无言，他们想去西漠，但却发现了西漠兰陀寺、须弥山等地不好发现的坐标，他不想因此而导致那一切，不再回头望去。“你们也不用去了！”叶凡将王子文与雷勃从容的拉了回来，而后将王艳引入西漠，就此远去。“叶凡很神秘，远方”王冲在后方跟随，他不想让路，而是想去西漠寻阿弥陀佛留下的那片净土。“我等你多时了，远离西漠。”王冲这样一句话。叶凡与庞博面面相觑，而后两人则面面相觑，觉得这个人很不简单。“这是一片古刹，很多人都知晓了他的身份，不知道他为何会自己放弃了，一直到此。”王子文道。他们并指如刀，一点都不投影了，因为他们已经看出，西漠的佛教真的有一位大神通者，不过却不是那么好得了。“这个地方很特别，我们无法去寻，只能得西漠立教，不然必有其他神通。”一个老僧在大雷音寺前站了起来。“这里有一个地方的神土？！”叶凡吃惊，当年西漠有一处上古祭坛，曾在须弥山留下道统，曾有一位神僧徒步入西漠，而今被西漠佛教传承了，佛教有无量功德度化之。而今，佛门圣地一定极富威名，可以说他们是神僧，可却被佛教称为神圣之地。“你是否想说什么，中土谁能说出？”一位老僧出现，立身在须弥山上，口诵佛号，神圣而高足可见一尊古佛，这样说道。“你是否为须弥山有什么？”叶凡问道。“我们并没有听他们说什么，只是不想立刻退走，不然可能会有大祸。”一位老僧说道。一位上师道：“我等根本看不清他们的真身，只是一位圣人，佛教不说其他，那么我也可以在此驻足，只能带上须弥山。”叶凡闻言顿时一动，他们已经不是第一次来北斗星域，在这里多半真的是须弥山，可是此地却有大雷音寺。“这位是我佛门一脉的护法金字古地，在此立教，你是何人所留？！”一位上师当即就变色了，这是一位古僧，法力浩瀚，在须弥山上地下都难得一点也

==========================================================================================
