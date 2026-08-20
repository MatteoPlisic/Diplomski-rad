# Plan prezentacije diplomskog rada

Cilj: 12-15 slajdova, ~15 min izlaganje + Q&A.

---

## Slajd 1 — Naslovna

- Naslov rada
- Ime i prezime, mentor
- Datum, fakultet

---

## Slajd 2 — Motivacija (zasto AMP?)

- Otpornost na antibiotike kao globalni problem
- Antimikrobni peptidi (AMP) kao alternativa: prirodni, siroki spektar, manja
  vjerojatnost rezistencije
- Problem: pronalaziti novog kandidata laboratorijski je sporo i skupo
- Rjesenje: generativni AI + scoring model

---

## Slajd 3 — Sto je REINVENT?

- Originalni REINVENT (AstraZeneca, 2018): RL framework za generiranje
  malih organskih molekula u SMILES formatu
- Komponente:
  - **Prior** — RNN (GRU) treniran na ChEMBL bazi (1.5M molekula)
  - **Agent** — kopija priora koja se fine-tuna RL-om
  - **Scoring function** — sto dobivene molekule cijene kao "dobre"
- Augmented-likelihood formula:
  ```
  target = prior_likelihood + sigma * score
  loss = (target - agent_likelihood)^2
  ```

---

## Slajd 4 — Adaptacija za peptide

- REINVENT defaultno generira male organske molekule, ne peptide
- Sto smo dodali:
  - Peptide klasifikator (PepFuNN)
  - Peptide filter u scoring pipeline
  - Normalizacija scorea
  - Modificirana formula augmented likelihooda (prior_weight)
  - Experience replay konfiguriran

**Slika:** dijagram pipelinea (SMILES -> RF -> filter -> norm -> loss)

---

## Slajd 5 — Scoring pipeline

Vizualno prikazati 4 koraka:

1. **RF AMP klasifikator** — Random Forest na 293 Mordred deskriptora,
   vraca P(AMP active) u [0, 1]
2. **Peptide filter** — PepFuNN klasifikacija; non-peptidi -> score 0
3. **Normalizacija** — `(score - 0.3) / (0.85 - 0.3)`, clip [0, 1]
4. **Augmented likelihood** — `0.8 * prior + 80 * score`

---

## Slajd 6 — Kako PepFuNN klasificira peptid?

- `peptideFromSMILES()` rastavlja molekulu duz peptidnih veza
- Svaki fragment mapira na biblioteku aminokiselina
- Vraca sekvencu tipa `G-A-F-X1726-K`
- Klasifikacija:
  - **PEPTIDE** — sve standardne L-AK
  - **PEPTIDE_NONCANONICAL** — neki "Xnnn" (npr. dOrn, X1668)
  - **NOT_PEPTIDE** — premalo ostataka

**Slika:** primjer SMILES -> rastavljanje -> sekvenca

---

## Slajd 7 — Varijante filtera

Tablica:

| Varijanta | min AK | max % nekanon. | bez 3+ istih | mekana kazna |
|---|---|---|---|---|
| A (basic) | 2 | - | - | - |
| B (strict) | 2 | - | da | - |
| C (strict_v2) | 4 | 25% | - | - |
| D (strict_v3) | 4 | 25% | da | - |
| E (strict_v4) | 4 | 25% | da | 0.85x |

Kratko obrazlozenje kazne svakog pravila:
- Min 4 AK: prirodni AMP-ovi su 10-50 AK
- Max 25% nekanon.: prakticnije sintetizirati
- Bez triple: izbjeci homopolimere (Cesto false-positive RF model)
- 0.85x kazna: pogurnuti agent prema potpuno standardnima

---

## Slajd 8 — Score progress: A varijanta x3

Tri grafa (avg score po koraku) jedan pored drugog.

Naglasak: **RL je stohastican** — od istih parametara dobivamo razlicite
distribucije. Zato treniramo vise puta.

Tablica:

| Run | Final avg score | % nekanonskih |
|---|---|---|
| A_basic_1 | ~0.5 | 24% |
| A_basic_2 | ~0.5 | 100% |
| A_basic_3 | ~0.5 | 64% |

---

## Slajd 9 — Score progress: stroze varijante

Grafovi za B, C, D, E na jednom slajdu.

Highlights:
- D postize najveci avg score (49 PEPTIDE / 50)
- E ima 0% nekanonskih (soft penalty radi!)
- Stroziji filter ne mora znaciti losiji score — naprotiv

---

## Slajd 10 — Warm-start eksperiment (D-from-C)

- Sto: D pravila, ali prior i pocetni agent = C-jev trenirani Agent.ckpt
- Hipoteza: brza konvergencija jer agent vec "zna" peptide
- Rezultat: ALI naslijedio je C-jevu sklonost nekanonskim (62% nekanonskih)
- Trade-off: brzina vs. exploration prostor

**Slika:** usporedba score-history krivulja D (od nule) i D-from-C

---

## Slajd 11 — Sequence logo: kako citati?

- X os: pozicija u peptidu (lijevo poravnano)
- Y os: frekvencija AK na toj poziciji
- Boja: kategorija (zelena = polarna, plava = bazicna, ...)
- Visina slova = visina vjerojatnosti

**Prosirena abeceda 20 + 12 nekanonskih:**

Mapiranje (grcka slova za nekanonske):
- α = dOrn, β = dDpr, γ = dDab, δ = Hse
- ε-μ = razne X-numbered iz PepFuNN biblioteke

---

## Slajd 12 — Sequence logo: usporedba varijanti

Side-by-side prikaz 2-3 logoa.

Predlazem:
- **C (strict_v2)** — vidi se mjesavina kanonskih i nekanonskih
- **D (strict_v3)** — gotovo sve kanonske, jasni motivi
- **E (strict_v4)** — cisti kanonski

Naglasak: agent uci konkretne pozicione preferencije (npr. Gly + Ser
na pocetku, Cys u sredini, itd.)

---

## Slajd 13 — Najbolje generirane molekule

Tablica top-5 sa svakog runa:

| Score | Sekvenca | Run |
|---|---|---|
| 0.94 | NCC...A-G-E-K... | D |
| 0.91 | ... | A |
| ... | | |

Mozda kemijske strukture 2-3 najboljih kao slika.

---

## Slajd 14 — Sto smo naucili

- **Peptide filter je ESENCIJALAN** — bez njega prior generira male
  organske molekule s visokim AMP score "false positive"
- **Score normalizacija** dramaticno ubrzava konvergenciju
- **Soft penalty (varijanta E)** = elegantnije od hard rejecta — agent ne
  treba istraziti "zabranjen" prostor, dovoljno mu je da ga blago
  destimuliras
- **Warm-start nasljeduje bias** — vazno pamtiti pri produkcijskim
  eksperimentima
- **RL je stohastican** — uvijek raditi vise runova istog setupa

---

## Slajd 15 — Daljnji rad

Idee za nastavak:

- **Seed memory** — pre-fill experience replay sa znanim AMP-ovima
- **Transfer learning prior** — fine-tunati prior na peptidnoj bazi prije RL
- **Vise scoring funkcija istovremeno** — npr. AMP + toksicnost + topivost
- **Eksperimentalna validacija** — sintetizirati top 5 i testirati antimikrobnu
  aktivnost u laboratoriju
- **Veci modeli** — Transformer umjesto GRU

---

## Slajd 16 — Q&A / Zahvala

- Hvala
- Slika prirodnih AMP-ova (npr. melittin, magainin) za vizualni kraj
- Kontakt email

---

## Tehnicki savjeti za prezentaciju

1. **Slike > tekst.** Svaki slajd ima sliku/graf/dijagram, ne samo bulleti.
2. **Avgust pricam o jednom konceptu po slajdu.** Ne nabacivati 5 ideja.
3. **Demonstracija po mogucnosti** — pokazi GUI ili memory file uzivo
   ako je 30s spustanje s desktopa.
4. **Ne zaboravi:**
   - Ucitati svih 8 sequence_logo.png prije prezentacije
   - Pripremiti score_progress.png za bar 4 runa
   - Demo run u rezervi ako te netko pita "ali kako to izgleda kad krene"
5. **Q&A pripremiti odgovore na:**
   - "Zasto Random Forest a ne deep network za scoring?"
   - "Kako bi se ovo prosirilo na druge bioloske aktivnosti?"
   - "Sto ako je RF model lazno pozitivan?"
   - "Jesu li ove molekule sintetizirajuce u laboratoriju?"
   - "Sto je sa kombinatornom eksplozijom?"
