# REINVENT za generiranje antimikrobnih peptida (AMP)

Diplomski rad — prilagodba generativnog modela **REINVENT** za domenu
aminokiselina. RNN agent uči generirati peptidne strukture s visokim
predviđenim AMP rezultatom (Random Forest model na Mordred deskriptorima),
uz dodatni filtar koji ga usmjerava prema valjanim peptidima.

---

## Brzi početak

```bash
# 1. okruženje (Python 3.10)
python -m venv .venv
.venv\Scripts\activate            # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. (jednokratno) popravi sklearn-kompatibilnost RF modela
python setup/fix_model.py         # stvara models/random_forest_model_amp_fixed.pkl

# 3. treniranje agenta (primjer: varijanta D, 250 koraka)
python main.py --scoring-function tanimoto --num-steps 250 ^
               --sigma 80 --prior-weight 0.8 --filter-mode strict_v3

# 4. analiza rezultata (pokretati IZ KORIJENA projekta)
python analysis/check_all_runs.py          # klasifikacija peptida (PepFuNN)
python analysis/summary_table.py           # zbirna tablica + graf nekanonskih
python analysis/sequence_logo.py           # sequence logo po runu
python analysis/compare_scores.py          # usporedni grafovi scora
```

> **Važno:** svi alati koriste putanje relativne na korijen projekta
> (`data/results`, `models/…`), pa ih uvijek pokreći iz korijena
> (`python analysis/<alat>.py`), a ne iz podfoldera.

Rezultati svakog treninga spremaju se u `data/results/run_<datum>/`.

---

## Struktura projekta

```
diplomski-rad/
│
├── main.py                  # CLI ulaz — parsira argumente i pokreće trening
├── train_agent.py           # RL petlja: prior + agent + scoring + peptide_filter
├── model.py                 # RNN (3× GRU) generativni model
├── data_structs.py          # Vocabulary, tokenizacija, Experience replay
├── scoring_functions.py     # funkcije vrednovanja (tanimoto = AMP RF model)
├── utils.py                 # Variable, seq_to_smiles, ...
├── vizard_logger.py         # logiranje težina tijekom treninga
│
├── analysis/                # alati za analizu rezultata (samostalni)
│   ├── check_all_runs.py            # PepFuNN klasifikacija svih runova
│   ├── check_memory_peptides_pepfunn.py  # klasifikacija najnovijeg runa
│   ├── summary_table.py             # master zbirna tablica + bar graf
│   ├── sequence_logo.py             # sequence logo (20 AK + 12 nekanonskih)
│   ├── compare_scores.py            # usporedni grafovi scora
│   ├── noncanonical_inventory.py    # popis svih nekanonskih AK
│   └── identify_runs.py             # heuristika za prepoznavanje runa
│
├── scripts/                 # PowerShell skripte za pokretanje varijanti
│   ├── run_variants.ps1  run_bc.ps1  run_d.ps1
│   ├── run_d_from_c.ps1  run_e.ps1  run_f.ps1
│
├── setup/
│   └── fix_model.py         # jednokratni popravak sklearn pickle kompatibilnosti
│
├── models/                  # AMP Random Forest modeli (.pkl)
│   ├── random_forest_model_amp.pkl         # original (sklearn 0.21)
│   └── random_forest_model_amp_fixed.pkl   # popravljen za sklearn 1.7
│
├── data/
│   ├── Prior.ckpt           # bazni ChEMBL RNN (~1.18M molekula)
│   ├── Voc                  # vokabular SMILES tokena (49)
│   └── results/             # izlazi treninga (run_<datum>_<varijanta>/)
│
├── requirements.txt
└── README.md
```

Pokretanje varijanti preko skripti (iz korijena):
```powershell
.\scripts\run_f.ps1          # varijanta F (strict_v5), 250 koraka
```

---

## Pipeline scoringa

```
score = scoring_function(SMILES)                          # RF model 0..1
score = score * peptide_filter(SMILES, mode)              # ne-peptidi -> 0
score = clip((score - 0.3) / (0.85 - 0.3), 0, 1)          # normalizacija
augmented_likelihood = 0.8 * prior_likelihood + 80 * score
loss = (augmented_likelihood - agent_likelihood)^2
```

Parametri zajednicki svim varijantama:

| param | vrijednost | znacenje |
|---|---|---|
| `--num-steps` | 250 | broj RL koraka |
| `--batch-size` | 64 | molekula po koraku |
| `--sigma` | 80 | tezina scora u augmented likelihoodu |
| `--prior-weight` | 0.8 | tezina priora (< 1.0 = vise istrazivanja) |
| `--experience` | 10 | top molekula iz memorije po koraku |

---

## Varijante filtra peptida (`--filter-mode`)

| Varijanta | mode | min AK | max % nekanon. | bez 3+ istih | kazna nekanon. |
|---|---|---|---|---|---|
| A | `basic` | 2 | — | — | — |
| B | `strict` | 2 | — | da | — |
| C | `strict_v2` | 4 | 25% | — | — |
| D | `strict_v3` | 4 | 25% | da | — |
| E | `strict_v4` | 4 | 25% | da | 0.85× |
| F | `strict_v5` | 4 | 25% | da | 0.5× |

Dodatno: **D-from-C** — varijanta D, ali prior i pocetni agent ucitani iz
prethodnog C runa (topli start). Pokrece se s `--prior` i `--agent` koji
pokazuju na `data/results/run_..._C_strict_v2/Agent.ckpt`.

---

## Klasifikacija peptida (PepFuNN)

`peptide_filter()` u `train_agent.py` koristi PepFuNN-ovu `peptideFromSMILES()`
da molekulu razloži po peptidnim vezama i mapira fragmente na biblioteku
aminokiselina. Verdiktima: PEPTIDE / PEPTIDE_NONCANONICAL / PEPTIDE_PARTIAL /
NOT_PEPTIDE. Filtar prihvaca samo PEPTIDE i PEPTIDE_NONCANONICAL (uz dodatna
ogranicenja po varijanti); sve ostalo dobiva score 0.

---

## Napomene za razvoj

- **Generator** je u korijenu (flat importi: `from model import RNN`), pa
  `python main.py` radi bez paketne strukture. `train_agent.py` se kopira u
  svaki `results/run_*/` radi reproducibilnosti.
- **Ograničenje duljine:** generator staje na 140 tokena (`model.py`,
  `sample()`), što odgovara peptidima od ~12–17 aminokiselina.
- **GPU:** trening automatski koristi CUDA ako je dostupna (testirano na
  RTX 3060, PyTorch 2.5.1+cu121).
