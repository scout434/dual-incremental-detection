# Status1 娑堣瀺瀹為獙璇存槑

杩欑粍娑堣瀺瀵瑰簲璁烘枃琛ㄤ腑 `Seq FT / Incremental Head / DuET Module / L_Distill / L_DC`
杩欏嚑椤圭粍浠躲€備负浜嗚妭鐪佹椂闂达紝闄?`00_no_seqft` 澶栵紝鍏朵綑瀹為獙閮藉鐢ㄥ凡缁忚缁冨ソ鐨?T1 checkpoint锛岀劧鍚庡彧璁粌 T2銆?
## 鍓嶇疆鏂囦欢

涓诲疄楠岀殑 T1 妯″瀷蹇呴』瀛樺湪锛?
```text
output/status1/main/task_1_voc_1_10_best.pt
```

GI 鍒嗘瘝鍙傝€冩ā鍨嬩篃闇€瑕佸瓨鍦細

```text
output/status1/ref_voc/task_1_voc_11_20_best.pt
output/status1/ref_clipart/task_1_clipart_1_10_best.pt
```

濡傛灉杩欎簺鏂囦欢涓嶅瓨鍦紝鍏堣繍琛岋細

```powershell
python status1\train_duet.py --config status1\configs\train.yaml
python status1\train_duet.py --config status1\configs\ref_voc.yaml
python status1\train_duet.py --config status1\configs\ref_clipart.yaml
```

## 娑堣瀺缁勫埆

```text
00_no_seqft
  Seq FT: no
  Incremental Head: no
  DuET Module: no
  L_Distill: no
  L_DC: no
  璇存槑锛氫笉澶嶇敤 T1锛屼粠 reference_full_head 寮€濮嬭缁?T2銆?
01_seqft
  Seq FT: yes
  Incremental Head: no
  DuET Module: no
  L_Distill: no
  L_DC: no
  璇存槑锛氬鐢?T1锛岀洿鎺ラ『搴忓井璋?T2锛屾棫绫昏涓嶅仛鍥炲～淇濇姢銆?
02_seqft_incremental_head
  Seq FT: yes
  Incremental Head: yes
  DuET Module: no
  L_Distill: no
  L_DC: no
  璇存槑锛氬鐢?T1锛孴2 鍚庢妸鏃х被鍒嗙被琛屼粠 T1 鍥炲～锛屾柊绫诲垎绫昏鏉ヨ嚜 T2銆?
03_seqft_incremental_head_duet
  Seq FT: yes
  Incremental Head: yes
  DuET Module: yes
  L_Distill: no
  L_DC: no
  璇存槑锛氬叡浜?backbone/neck 璧?DuET 铻嶅悎锛宧ead 琛岀骇鍥炲～銆?
04_seqft_incremental_head_duet_distill
  Seq FT: yes
  Incremental Head: yes
  DuET Module: yes
  L_Distill: yes
  L_DC: no
  璇存槑锛氬湪 03 鍩虹涓婂姞鍏ユ棫妯″瀷钂搁銆?
05_full
  Seq FT: yes
  Incremental Head: yes
  DuET Module: yes
  L_Distill: yes
  L_DC: yes
  璇存槑锛氬畬鏁存柟娉曘€?```

## 鐢熸垚閰嶇疆

鍙敓鎴愯缁冮厤缃拰璇勪及閰嶇疆锛屼笉寮€濮嬭缁冿細

```powershell
python status1\run_ablation.py --all --materialize-only
```

鐢熸垚鍚庝細寰楀埌锛?
```text
status1/configs/ablations/00_no_seqft.yaml
status1/configs/ablations/00_no_seqft_eval.yaml
...
status1/configs/ablations/05_full.yaml
status1/configs/ablations/05_full_eval.yaml
status1/configs/ablations/index.json
```

## 鍗曠嫭杩愯鏌愪竴缁?
鍙窇 `00_no_seqft`锛?
```powershell
python status1\run_ablation.py --name 00_no_seqft
python status1\eval_paper_metrics.py --plan status1\configs\ablations\00_no_seqft_eval.yaml
```

鍙窇 `01_seqft`锛?
```powershell
python status1\run_ablation.py --name 01_seqft
python status1\eval_paper_metrics.py --plan status1\configs\ablations\01_seqft_eval.yaml
```

鍙窇 `02_seqft_incremental_head`锛?
```powershell
python status1\run_ablation.py --name 02_seqft_incremental_head
python status1\eval_paper_metrics.py --plan status1\configs\ablations\02_seqft_incremental_head_eval.yaml
```

鍙窇 `03_seqft_incremental_head_duet`锛?
```powershell
python status1\run_ablation.py --name 03_seqft_incremental_head_duet
python status1\eval_paper_metrics.py --plan status1\configs\ablations\03_seqft_incremental_head_duet_eval.yaml
```

鍙窇 `04_seqft_incremental_head_duet_distill`锛?
```powershell
python status1\run_ablation.py --name 04_seqft_incremental_head_duet_distill
python status1\eval_paper_metrics.py --plan status1\configs\ablations\04_seqft_incremental_head_duet_distill_eval.yaml
```

鍙窇 `05_full`锛?
```powershell
python status1\run_ablation.py --name 05_full
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

## 涓€娆¤窇瀹屽叏閮ㄦ秷铻?
璁粌鍏ㄩ儴 6 缁勶細

```powershell
python status1\run_ablation.py --all
```

鐒跺悗渚濇璇勪及锛?
```powershell
python status1\eval_paper_metrics.py --plan status1\configs\ablations\00_no_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\01_seqft_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\02_seqft_incremental_head_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\03_seqft_incremental_head_duet_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\04_seqft_incremental_head_duet_distill_eval.yaml
python status1\eval_paper_metrics.py --plan status1\configs\ablations\05_full_eval.yaml
```

姣忕粍杈撳嚭淇濆瓨鍦細

```text
output/status1/ablations/<ablation_name>/metrics.json
output/status1/ablations/<ablation_name>/rai_metrics.json
```

## 鎸囨爣

姣忕粍閮借绠楋細

```text
Avg RI = final 鍦?VOC[1:10] 涓婄殑 mAP / T1 妯″瀷鍦?VOC[1:10] 涓婄殑 mAP
Avg GI = mean(
  final 鍦?VOC[11:20] 涓婄殑 mAP / VOC[11:20] 鍙傝€冩ā鍨?mAP,
  final 鍦?Clipart[1:10] 涓婄殑 mAP / Clipart[1:10] 鍙傝€冩ā鍨?mAP
)
RAI = (Avg RI + Avg GI) / 2
```


