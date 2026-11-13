# data/

Local datasets. Everything here except this file is gitignored.

Expected files:

- `flores_urd.txt`, `flores_eng.txt`: FLORES-200 devtest, `urd_Arab` and `eng_Latn`, one sentence per line, aligned line by line (1,012 lines each). Create them with `zubaan data flores`.
- `flores_meta.json`: where the FLORES files came from (source, archive checksum or dataset revision, date).
- `urdu_train.txt`: Urdu text for learning new merges (stage 2).
- `urdu_cpt.txt`: Urdu text for continued pretraining (stage 3).

Check the license of every source before use and record it here.

## Licenses

- FLORES-200 (Meta, `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`) and FLORES+ (`openlanguagedata/flores_plus`, gated on Hugging Face): CC BY-SA 4.0. Not redistributed in this repository; reports credit the source.
