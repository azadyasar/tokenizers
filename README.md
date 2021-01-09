# tokenizers
This repository contains several unsupervised text tokenizers for the Turkish language trained on two large datasets:
1) Turkish part of the [OSCAR](https://oscar-corpus.com/) corpus that contains ~30GB of raw text
2) +4000 books (~5GB)

The repository has two folders: (1) **cased** contains models trained on the cased version of the corpus, (2) **lowercased**, on the other hand, contains models trained on the lowercased version of the corpus. We also provide models trained with different frameworks.

## [sentencepiece](https://github.com/google/sentencepiece)
 Files with the **sp** prefix are trained with the sentencepiece framework. This framework also provides an additional file for each model that ends with *vocab* and contains all the tokens in raw text format.

 Files have the following naming convention
 ```
 sp_bpe_(lower)<vocab_size>_<sentence_size>
 ```
 where *sentence size* represents the number of sentences used to train the model and *lower* only exists when the model is lowercased.

 ### Hands On
 In order to install the sentencepiece module, issue the following command    
 ```pip install sentencepiece```.

``` python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='lowercased/sp_bpe_lower40k_50M.model')
print(sp.encode_as_pieces("Gelecekte, konuşma sentezleyiciler ve konuşma anlama alanındaki gelişmeler ve makine-insan iletişiminin gelişmesi, insanın makineden beklentilerini yükseltecektir.".lower()))
# => ['▁gelecekte', ',', '▁konuşma', '▁sentez', 'leyiciler', '▁ve', '▁konuşma', '▁anlama', '▁alanındaki', '▁gelişmeler', '▁ve', '▁makine', '-', 'insan', '▁iletişim', 'inin', '▁gelişmesi', ',', '▁insanın', '▁makin', 'eden', '▁beklentilerini', '▁yüksel', 'tecek', 'tir', '.']
print(sp.encode("Gelecekte, konuşma sentezleyiciler ve konuşma anlama alanındaki gelişmeler ve makine-insan iletişiminin gelişmesi, insanın makineden beklentilerini yükseltecektir.".lower()))
# => [9334, 39762, 3935, 17962, 37042, 38, 3935, 4222, 13612, 5668, 38, 6612, 39776, 10661, 2627, 149, 11346, 39762, 2489, 2371, 290, 23596, 1591, 13135, 177, 39758]
sp.piece_to_id("gelecek")
# => 34536
```

## [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe)
Models that have the **yttm** prefix are trained with the YouTokenToMe framework. YTTM models, as opposed to sentencepiece models, are trained on the whole corpus (187M sentences) without sampling.
### Hands On
Install the module by running ```pip install youtokentome``` in order to use the tokenizer.
``` python
import youtokentome as yttm
model = yttm.BPE('lowercased/yttm_lower50K.model')
print(model.encode("Gelecekte, konuşma sentezleyiciler ve konuşma anlama alanındaki gelişmeler ve makine-insan iletişiminin gelişmesi, insanın makineden beklentilerini yükseltecektir.".lower(), output_type=yttm.OutputType.SUBWORD))
# => ['▁gelecek', 'te,', '▁konuşma', '▁sentez', 'leyiciler', '▁ve', '▁konuşma', '▁anlama', '▁alanındaki', '▁gelişmeler', '▁ve', '▁makine', '-', 'insan', '▁iletişim', 'inin', '▁geliş', 'mesi,', '▁insanın', '▁makin', 'eden', '▁beklentilerini', '▁yüksel', 'tecek', 'tir.']
print(model.encode("Gelecekte, konuşma sentezleyiciler ve konuşma anlama alanındaki gelişmeler ve makine-insan iletişiminin gelişmesi, insanın makineden beklentilerini yükseltecektir.".lower(), output_type=yttm.OutputType.ID))
# => [2358, 3930, 3982, 18637, 45797, 385, 3982, 4576, 14738, 5470, 385, 7622, 42, 12117, 3732, 498, 1066, 4156, 3070, 2774, 643, 25767, 2023, 14498, 795]
```

### Appendix
For more information regarding the frameworks refer to the following repositories:
- [sentencepiece](https://github.com/google/sentencepiece)
- [YouTokenToMe](https://github.com/VKCOM/YouTokenToMe)

and for the BPE and subwords:
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [BPE-Dropout: Simple and Effective Subword Regularization](https://arxiv.org/abs/1910.13267)
