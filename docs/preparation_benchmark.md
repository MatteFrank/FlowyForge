# Benchmark preparazione COLLIDE e partizioni

La prima sezione conserva le misure del Task 1. Le misure della successiva
[ottimizzazione strict](#ottimizzazione-strict-fingerprint-esatti) sono riportate
in fondo: riguardano le sole partizioni e non vanno confuse con i tempi dell'intero
workflow preparazione + audit + split.

Misure locali dell'implementazione in working tree basata su `main` `9e1a380`.
Baseline: **codice originale `9e1a380`**, esportato con `git archive` e invocato
in un subprocess con il proprio `src` in `PYTHONPATH`: prepare seriale seguito
da creazione strict delle partizioni. Il batch di creazione delle partizioni è
2048 anche nella baseline, rispettando la rimozione del cap di quattro righe.
La working tree, i dataset e gli output esistenti non sono stati sostituiti.

Hardware: **Apple M4 Pro, 14 CPU logiche**, macOS 26.5.2, Python 3.13.5.
Ogni caso usa un processo nuovo; `OMP_NUM_THREADS`, `MKL_NUM_THREADS` e
`OPENBLAS_NUM_THREADS` sono impostati a 1; i worker limitano anche Arrow/Torch.
La misura comprende il pool e tutti gli output della preparazione, esclude gli
import Python. Una sola ripetizione, cache filesystem non controllata. Differenze
piccole non sono evidenza di scaling. RSS è un high-water mark di vita del
processo; la somma dei picchi è un limite superiore conservativo, **non** la
misura simultanea dell'intero job. I tempi di fase paralleli sono somme dei worker,
non vanno sommati al wall time. Fingerprinting è una sottofase delle partizioni;
le verifiche SHA della baseline sono conteggiate anche dentro export/audit.
La verifica sorgente legacy non è isolata (`n/a`); i suoi hash sono misurati nella
colonna «Legacy file SHA». Zero per fingerprint fast significa nessuna esecuzione.

Non è stato ottenuto 10×: sui dati reali la migliore osservazione è **2,98×**
rispetto al workflow precedente (17,588 → 5,902 s). Con solo due file il
parallelismo utile è al massimo due; la differenza fra 2/4/8 worker richiesti
riflette anche rumore e cache. L'export e l'audit fisico restano necessari.
Sui piccoli file sintetici lo startup dei processi può superare il vantaggio
computazionale. **Nessuna misura sul cluster CNAF, sui 20 file reali TERABIT o
sul pilot di 50.000 eventi per processo**; nessun training reale in questo
benchmark. Il target va rimisurato sul carico effettivo, non estrapolato.

## Dati e comandi eseguiti

- Sintetici: `make_tiny_parquet_sample(..., events_per_class=640, files_per_class=10,
  seed=42)`, sei collections in entrambe le viste, 20 file. Selezione 512 per
  processo, seed 42, batch export 64; 1.488.012 byte di input.
- Reali: i due shard esistenti di `data/processed/collide_pilot`, 5.000 righe
  ciascuno, provenienza HF persistente dalla revisione
  `e67bf675673b371ee2303b3db00c59ef42375162`, processi ttbar semileptonico e W+jets.
  **Re-export di un campione da file selezionati**, non preparazione dagli
  originali raw e non campione uniforme di COLLIDE-1m. Selezione 64 per processo,
  seed 42, batch export 64; 1.702.984.620 byte di input. Fast riusa gli origin già
  presenti senza ricalcolare il digest dell'originale. Il costo del digest
  iniziale per file raw è invece incluso nel caso sintetico.
- In tutti i casi: frazioni 60/20/20, medesime identità/etichette/appartenenze
  verificate automaticamente. I checksum di manifest delle policy restano distinti.

```bash
python3 scripts/benchmark_collide_preparation.py --config outputs/preparation_benchmark_inputs/synthetic.yaml --output-dir outputs/preparation_benchmark_synthetic --legacy-revision 9e1a380
python3 scripts/benchmark_collide_preparation.py --config outputs/preparation_benchmark_inputs/real.yaml --output-dir outputs/preparation_benchmark_real --legacy-revision 9e1a380
```

Per riprodurre, creare due YAML local selected-file con quei parametri e con
`model_config` puntato a `configs/models/foundation_tiny.yaml` (percorso assoluto,
o relativo alla directory del YAML); adattare `source.processes` ai propri file.
Gli output/config generati e i JSON/CSV completi rimangono sotto `outputs/`,
esclusi dal versionamento. Il benchmark rifiuta una directory di destinazione
esistente; usare un nome diverso per ogni nuova invocazione. Il README contiene
il comando direttamente applicabile al config TERABIT e le opzioni Slurm.

## Sintetici: risultati integrali

Platform: macOS-26.5.2-arm64-arm-64bit-Mach-O; CPU: arm; logical CPUs: 14.
Input: 20 files / 1,488,012 bytes; 512 selected events/process.
Legacy revision: `9e1a380`. Membership checksum: `b947c1f3399eb92fcb9d48cca3113a6d94e0976d294e74386a568b0ccce0655f`.

Local cache order uncontrolled; one fresh process/case; phase sums are worker elapsed times, include nested subphases; RSS lifetime peaks, summed only as a conservative upper bound, not measured concurrent peak. No training or scientific metrics.

| Policy | Workers | Repeat | Wall s | Events/s | Parallel speedup | Legacy speedup | Parent peak MiB | Worker max MiB | RSS upper bound MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 1 | 1 | 1.435 | 713.7 | 1.00× | 1.00× | 255.3 | 0.0 | 255.3 |
| fast | 1 | 1 | 1.254 | 816.6 | 1.00× | 1.14× | 253.5 | 252.8 | 253.5 |
| fast | 2 | 1 | 1.487 | 688.7 | 0.84× | 0.96× | 208.7 | 251.2 | 710.9 |
| fast | 4 | 1 | 1.329 | 770.4 | 0.94× | 1.08× | 210.1 | 252.0 | 1217.4 |
| fast | 8 | 1 | 1.881 | 544.4 | 0.67× | 0.76× | 206.8 | 233.1 | 2054.0 |
| strict | 1 | 1 | 1.416 | 723.0 | 1.00× | 1.01× | 254.3 | 253.0 | 254.3 |
| strict | 2 | 1 | 1.576 | 649.7 | 0.90× | 0.91× | 210.5 | 251.9 | 713.7 |
| strict | 4 | 1 | 1.325 | 772.7 | 1.07× | 1.08× | 209.2 | 252.0 | 1216.3 |
| strict | 8 | 1 | 1.372 | 746.6 | 1.03× | 1.05× | 209.7 | 253.1 | 2227.4 |

Worker phase totals (seconds; fingerprint time is included in partition collection/generation):

| Policy | Workers | Discovery | Source verification | Export | Partition collection | Assignment/save | Event fingerprints | Validation | Output SHA | Legacy file SHA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 1 | 0.018 | n/a | 0.342 | 0.000 | 0.155 | 0.034 | 0.882 | 0.000 | 0.005 |
| fast | 1 | 0.024 | 0.010 | 0.315 | 0.016 | 0.032 | 0.000 | 0.828 | 0.000 | 0.000 |
| fast | 2 | 0.023 | 0.011 | 0.513 | 0.016 | 0.030 | 0.000 | 0.814 | 0.000 | 0.000 |
| fast | 4 | 0.024 | 0.012 | 0.975 | 0.017 | 0.030 | 0.000 | 0.839 | 0.000 | 0.000 |
| fast | 8 | 0.023 | 0.024 | 2.829 | 0.032 | 0.030 | 0.000 | 1.362 | 0.000 | 0.000 |
| strict | 1 | 0.029 | 0.011 | 0.346 | 0.106 | 0.036 | 0.034 | 0.853 | 0.001 | 0.000 |
| strict | 2 | 0.025 | 0.011 | 0.526 | 0.105 | 0.036 | 0.034 | 0.824 | 0.001 | 0.000 |
| strict | 4 | 0.024 | 0.012 | 1.018 | 0.108 | 0.035 | 0.034 | 0.836 | 0.001 | 0.000 |
| strict | 8 | 0.025 | 0.015 | 1.946 | 0.113 | 0.036 | 0.036 | 0.877 | 0.002 | 0.000 |

## COLLIDE locale: risultati integrali

Platform: macOS-26.5.2-arm64-arm-64bit-Mach-O; CPU: arm; logical CPUs: 14.
Input: 2 files / 1,702,984,620 bytes; 64 selected events/process.
Legacy revision: `9e1a380`. Membership checksum: `40265e6b87f55c61a48d806c0675a650d337b406627a23378008711d0179cde1`.

Local cache order uncontrolled; one fresh process/case; phase sums are worker elapsed times, include nested subphases; RSS lifetime peaks, summed only as a conservative upper bound, not measured concurrent peak. No training or scientific metrics.

| Policy | Workers | Repeat | Wall s | Events/s | Parallel speedup | Legacy speedup | Parent peak MiB | Worker max MiB | RSS upper bound MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 1 | 1 | 17.588 | 7.3 | 1.00× | 1.00× | 1029.8 | 0.0 | 1029.8 |
| fast | 1 | 1 | 9.803 | 13.1 | 1.00× | 1.79× | 528.2 | 528.2 | 528.2 |
| fast | 2 | 1 | 6.398 | 20.0 | 1.53× | 2.75× | 270.5 | 529.9 | 1306.4 |
| fast | 4 | 1 | 5.902 | 21.7 | 1.66× | 2.98× | 262.3 | 506.7 | 1260.4 |
| fast | 8 | 1 | 6.536 | 19.6 | 1.50× | 2.69× | 286.9 | 539.0 | 1350.9 |
| strict | 1 | 1 | 16.837 | 7.6 | 1.00× | 1.04× | 597.0 | 597.0 | 597.0 |
| strict | 2 | 1 | 9.399 | 13.6 | 1.79× | 1.87× | 262.5 | 583.1 | 1374.4 |
| strict | 4 | 1 | 9.799 | 13.1 | 1.72× | 1.79× | 273.8 | 495.3 | 1255.8 |
| strict | 8 | 1 | 9.680 | 13.2 | 1.74× | 1.82× | 260.5 | 505.6 | 1264.4 |

Worker phase totals (seconds; fingerprint time is included in partition collection/generation):

| Policy | Workers | Discovery | Source verification | Export | Partition collection | Assignment/save | Event fingerprints | Validation | Output SHA | Legacy file SHA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy | 1 | 0.040 | n/a | 7.322 | 0.000 | 6.669 | 2.429 | 2.512 | 0.000 | 0.976 |
| fast | 1 | 0.065 | 0.027 | 7.185 | 0.008 | 0.005 | 0.000 | 2.486 | 0.000 | 0.000 |
| fast | 2 | 0.063 | 0.032 | 7.550 | 0.008 | 0.005 | 0.000 | 2.918 | 0.000 | 0.000 |
| fast | 4 | 0.062 | 0.030 | 7.399 | 0.008 | 0.005 | 0.000 | 2.557 | 0.000 | 0.000 |
| fast | 8 | 0.063 | 0.030 | 7.563 | 0.008 | 0.005 | 0.000 | 3.454 | 0.000 | 0.000 |
| strict | 1 | 0.067 | 0.732 | 7.067 | 6.319 | 0.005 | 2.428 | 2.604 | 0.017 | 0.000 |
| strict | 2 | 0.061 | 0.718 | 7.439 | 6.369 | 0.006 | 2.450 | 2.609 | 0.018 | 0.000 |
| strict | 4 | 0.064 | 0.732 | 7.812 | 6.445 | 0.006 | 2.496 | 2.599 | 0.021 | 0.000 |
| strict | 8 | 0.075 | 0.774 | 7.838 | 6.246 | 0.006 | 2.431 | 2.582 | 0.021 | 0.000 |

## Verifiche della pipeline

`python3 -m pytest tests/ -q`: **173 passed**, 552 warning di deprecazione
Matplotlib/Pillow, 197,38 s nella suite finale. I due test inizialmente falliti
fallivano anche sul codice originale `9e1a380`: assumevano ancora uguali epoche
default supervised/downstream e il vecchio percorso del config pilot. Sono stati
aggiornati alle configurazioni presenti, senza cambiare i parametri scientifici.

Gli smoke sintetici coprono entrambe le policy, FullReco/L1T, pretraining di una
sola epoca CPU, export encoder, scratch/linear probe/fine-tuning, report di
confronto, resume fast, appartenenze uguali, test escluso dal pretraining e uso
finale del test downstream. I test controllano inoltre assenza di hashing dei
payload fast, protezioni strict, proiezione delle sole colonne di provenienza,
worker 1/2/4/8, audit aggregato e cleanup dopo errore del worker.

Verifica reale aggiuntiva, senza training: una copia dell'export legacy del
benchmark (128 eventi totali) è stata usata con `--stage partitions` fast,
2 worker. Il nuovo manifest ha split **80/24/24**, identità/etichette/appartenenze
esattamente uguali al manifest strict precedente e SHA del manifest distinto.
Due `run_classification.py --stage inspect --device cpu` hanno letto FullReco e
L1T, 64 eventi di audit per vista. Avvisi attesi: unità assunte, PUPPIPart
troncate dal reader e Charge opzionale mancante per Electron/MuonTight.
Artefatti locali: `outputs/preparation_existing_sample_smoke/`,
`outputs/preparation_inspect_fullreco/`, `outputs/preparation_inspect_l1t/`.
Il campione originale sotto `data/processed/collide_pilot` è rimasto intatto.

## Ottimizzazione strict: fingerprint esatti

Misure del 26 novembre 2026, working tree basata su `c336c07`; confronto col
codice originale `9e1a380` e col Task 1 `c336c07` (implementazione `05a217e`).
Stesso Apple M4 Pro / 14 CPU logiche, filesystem locale, Python 3.13.5 del
benchmark precedente; NumPy 2.3.1 e PyArrow 24.0.0. Nessun training scientifico
o job CNAF eseguito.

Qui si misura **create + save del manifest strict**, su input già presenti,
seed 42, batch Arrow 64, frazioni 60/20/20. Ogni caso ha un processo nuovo e
le stesse limitazioni dei thread; startup del pool incluso, import iniziale
Python escluso. Due ripetizioni, in ordine originale/Task1/ottimizzato; cache
filesystem non controllata, nessuna statistica inferenziale. Nelle tabelle:
mediana wall/CPU/eventi-s, massimo RSS osservato nelle due ripetizioni. CPU
comprende i figli terminati; RSS totale è la somma dei massimi, non un picco
simultaneo. Originale è solo seriale: non viene simulato un suo parallelismo.

Input distinti, ciascuno identico per tutte le varianti confrontate:

- **Reali, 2 shard**: 128 eventi, 64/processo, 37.525.186 byte; export già
  esistente `outputs/preparation_existing_sample_smoke`, ttbar semileptonico e
  W+jets. Provenienza HF conservata dal campione a file selezionati descritto
  sopra. Non è un campione uniforme dell'intero COLLIDE.
- **Reali, 20 shard**: gli stessi 128 eventi, riscritti in 10 shard/processo
  con `pyarrow.parquet.write_table`, senza cambiare valori, schema o origini;
  36.946.038 byte. Serve a osservare il pool, **non rappresenta 20 file originali
  completi o il carico di produzione CNAF**.
- **Sintetici, 20 file**: `make_tiny_parquet_sample(events_per_class=640,
  files_per_class=10, seed=42)`, 1.280 eventi, 1.488.012 byte. Sono gli input
  del benchmark precedente, questa volta senza selezionare solo 512/processo.

Tutti i casi confrontati producono **lo stesso SHA256 dell'intero manifest
strict per lo stesso input**: si verificano anche contenuti, origini, label e
appartenenze, non solo conteggi. Due ripartizioni fisiche dei file possono avere
SHA di manifest diversi anche con gli stessi eventi. I JSON/CSV, manifest e
log per caso restano sotto `outputs/strict_optimization/`, esclusi da Git.

### Reali: due shard

| Codice | Worker | Wall s | CPU s | Eventi/s | Speedup/originale | RSS parent MiB | RSS totale, limite sup. MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 1 | 6.764 | 6.656 | 18.9 | 1.00× | 935.3 | 935.3 |
| task1 | 1 | 6.605 | 6.600 | 19.4 | 1.02× | 1042.8 | 1042.8 |
| task1 | 2 | 4.111 | 8.016 | 31.1 | 1.65× | 205.3 | 2006.9 |
| task1 | 4 | 4.075 | 8.022 | 31.4 | 1.66× | 205.3 | 2006.8 |
| task1 | 8 | 4.179 | 8.115 | 30.6 | 1.62× | 205.3 | 1939.1 |
| optimized | 1 | 3.472 | 3.390 | 36.9 | 1.95× | 734.4 | 734.4 |
| optimized | 2 | 2.508 | 4.933 | 51.0 | 2.70× | 205.0 | 1501.1 |
| optimized | 4 | 2.551 | 5.025 | 50.2 | 2.65× | 205.8 | 1505.8 |
| optimized | 8 | 2.519 | 4.963 | 50.8 | 2.68× | 204.8 | 1496.5 |

Manifest strict: `8f5d231553239415f0534963e51fbc1d162a51dce1725ff33a43c9df6b2a980b`. Dati completi: `outputs/strict_optimization/real_b64_v3/strict_benchmark.{json,csv,md}`.

### Reali: venti shard ricavati dagli stessi eventi

| Codice | Worker | Wall s | CPU s | Eventi/s | Speedup/originale | RSS parent MiB | RSS totale, limite sup. MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 1 | 6.639 | 6.458 | 19.3 | 1.00× | 377.4 | 377.4 |
| task1 | 1 | 6.642 | 6.575 | 19.3 | 1.00× | 362.0 | 362.0 |
| task1 | 2 | 4.196 | 8.032 | 30.5 | 1.58× | 206.3 | 906.2 |
| task1 | 4 | 2.745 | 9.909 | 46.7 | 2.42× | 205.3 | 1372.5 |
| task1 | 8 | 2.117 | 14.350 | 60.5 | 3.14× | 205.8 | 2362.3 |
| optimized | 1 | 3.530 | 3.440 | 36.3 | 1.88× | 405.8 | 405.8 |
| optimized | 2 | 2.679 | 5.182 | 47.8 | 2.48× | 205.5 | 992.5 |
| optimized | 4 | 2.098 | 7.519 | 61.6 | 3.16× | 204.7 | 1694.7 |
| optimized | 8 | 1.942 | 13.424 | 66.3 | 3.42× | 205.9 | 2863.1 |

Manifest strict: `b65b49a2e11369fbf03467976d9a0ba351a3acaecd291a9598199193fe555f11`. Dati completi: `outputs/strict_optimization/real20_b64/strict_benchmark.{json,csv,md}`.

### Sintetici: venti piccoli file

| Codice | Worker | Wall s | CPU s | Eventi/s | Speedup/originale | RSS parent MiB | RSS totale, limite sup. MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 1 | 0.216 | 0.203 | 5996.2 | 1.00× | 215.1 | 215.1 |
| task1 | 1 | 0.211 | 0.209 | 6074.9 | 1.02× | 214.9 | 214.9 |
| task1 | 2 | 0.819 | 1.547 | 1564.9 | 0.26× | 208.3 | 628.2 |
| task1 | 4 | 0.840 | 3.054 | 1527.3 | 0.26× | 208.4 | 1045.8 |
| task1 | 8 | 1.136 | 7.562 | 1135.4 | 0.19× | 205.9 | 1772.3 |
| optimized | 1 | 0.213 | 0.210 | 6018.0 | 1.01× | 214.9 | 214.9 |
| optimized | 2 | 0.816 | 1.547 | 1569.6 | 0.26× | 207.9 | 627.1 |
| optimized | 4 | 0.820 | 3.048 | 1560.9 | 0.26× | 207.1 | 1047.7 |
| optimized | 8 | 1.013 | 7.063 | 1263.9 | 0.21× | 206.1 | 1887.0 |

Manifest strict: `d619ba2e7027397b25a458fead2ddd9c4e7a6d9c3c3bdb9bd03f6510e3d31f70`. Dati completi: `outputs/strict_optimization/synthetic20_b64_final/strict_benchmark.{json,csv,md}`.

### Profilo e limiti del miglioramento

Sul caso reale a due shard, mediana seriale in secondi. Le fasi sono inclusive,
non una scomposizione additiva perfetta; il salvataggio include una nuova
validazione e serializzazione. `n/a` indica una fase non isolata nell'originale.

| Fase | Originale | Task 1 | Ottimizzato |
| --- | ---: | ---: | ---: |
| SHA256 dei file | 0.01812 | 0.01598 | 0.01565 |
| Lettura/decompressione/RecordBatch | 0.29649 | 0.28001 | 0.27614 |
| RecordBatch → Table | n/a | 0.00063 | 0.00004 |
| Arrow → Python, allocazioni nested incluse | 3.90313 | 3.74031 | 0.73615 |
| JSON canonico + UTF-8 per evento | 2.31746 | 2.34704 | 2.32991 |
| SHA256 per evento | 0.07095 | 0.07071 | 0.07070 |
| Schema | 0.00089 | 0.00417 | 0.00234 |
| Inventario, ordinamento globale e duplicati | n/a | 0.00050 | 0.00041 |
| Assemblaggio/split/validazione iniziale | 0.00304 | 0.00197 | 0.00191 |
| Validazioni del manifest | 0.00281 | 0.00260 | 0.00272 |
| Salvataggio del manifest | 0.00297 | 0.00344 | 0.00386 |

Lettura/decompressione e costruzione RecordBatch sono dentro un'unica chiamata
nativa `next(iter_batches)`: il profiler Python non le separa in modo affidabile.
Analogamente `to_pylist()` comprende creazione di liste/dizionari/scalari; la
nuova fase Arrow→Python comprende buffer NumPy, bitmap di validità e ricostruzione
nested. Il profilo originale dello schema misura JSON/hash; in Task 1/ottimizzato
include anche normalizzazione dei field. Inventario/sort/duplicati comprende la
scoperta dei file e il merge nel parent, non lo startup/attesa del pool.

Per i due shard: 37.525.186 byte Parquet, **81.349.700 byte Arrow** e
**212.313.536 byte JSON canonico** elaborati; manifest di circa 104 KB. Non sono
byte misurati sul dispositivo fisico: page cache, footer e letture del filesystem
non sono conteggiati come traffico hardware. I wrapper di misura introducono
un piccolo overhead. I record trasferiti dai worker contengono solo ID, origine
e digest, non oggetti fisici o JSON raw. Il merge è sotto 1 ms su questo campione:
non giustifica nuovi file temporanei e un secondo protocollo di fusione. Restano
al massimo N job/risultati in-flight; l'inventario finale cresce con gli eventi,
come prima.

**10× non raggiunto.** Sui due shard il guadagno seriale è 1,95× rispetto
all'originale e 1,90× rispetto al Task 1; a due worker è 2,70× rispetto
all'originale e 1,64× rispetto al Task 1 a due worker. Arrow→Python migliora
circa 5,3×, ma il JSON invariato resta circa 2,33 s, il 67% del wall seriale
ottimizzato. Con due file il pool non può usarne più di due contemporaneamente.
Anche eliminando gli altri costi, quel solo JSON lascia un tetto di circa 5,8×
rispetto all'originale seriale su due worker.

Sui venti shard reali piccoli si osserva **3,42×** rispetto all'originale
seriale con otto worker, ma solo **1,09×** rispetto al Task 1 a otto worker:
startup/import dei processi, contesa e I/O riducono il beneficio della conversione.
Quattro worker consumano meno CPU/RAM e danno un wall vicino a otto. Non si può
estrapolare questo dato ai venti file completi HPC; su file più grandi startup
ha un peso diverso. Non viene sostituito il JSON con un serializer che cambi
rappresentazione dei float o formati dei fingerprint per inseguire il target.

Sui sintetici minuscoli l'inizializzazione del percorso numerico era più costosa
del vecchio `to_pylist`: la misura esplorativa rilevava 0,375 s seriali contro
0,212 s originali. Il codice finale conserva `to_pylist` per porzioni fino a
64 KiB; la tabella sopra è stata rimisurata dopo questa correzione. Per pochi
millisecondi di lavoro/file, usare un solo worker evita il costo del pool.

La verifica completa degli export passa da tre hash del payload (inventario +
una lettura per vista) a **due**: hash iniziale condiviso e hash finale obbligatorio.
Un test conta le chiamate e modifica il file a parità di dimensione/mtime e anche
con metadata artificialmente invariati: il digest finale deve fallire. Nessuna
cache permanente, nessun riuso tra operazioni o epoche; i controlli strict del
reader e del binding non sono alleggeriti. La verifica del sorgente originale
rimane separata. Nei vecchi tempi di preparazione l'hash finale è ora compreso
nella fase di audit, non solo in `output_sha256_seconds`.

### Riproduzione e raccomandazioni HPC

Sweep aggiuntivo sullo stesso campione reale a due shard, codice finale, un
worker, due ripetizioni. Manifest identico anche fra tutti questi batch:

| Righe/batch richieste | Wall mediano s | RSS parent massimo MiB |
| ---: | ---: | ---: |
| 1 | 3,727 | 309,7 |
| 16 | 3,425 | 522,1 |
| 64 | 3,472 | 734,4 |
| 256 | 3,505 | 708,6 |

Ogni file contiene solo 64 righe: la riga 256 **non** misura un batch pieno da
256 eventi. 16 e 64 hanno tempi vicini; 16 riduce il picco su questi dati. La
porzione Python limita l'espansione, ma la RAM dipende ancora dal batch Arrow.
Per memoria stretta provare 16; 256 richiede una misura su file più grandi.

Comando eseguito per ciascun valore `B` in 1, 16, 256 (64 è già nel confronto):

```bash
python3 scripts/benchmark_collide_preparation.py --partition-source outputs/preparation_existing_sample_smoke --output-dir outputs/strict_optimization/batch_B --batch-size B --workers 1 --repeats 2
```

Sostituire `B` con il numero, anche nel nome dell'output.

```bash
python3 scripts/benchmark_collide_preparation.py --partition-source outputs/preparation_existing_sample_smoke --output-dir outputs/strict_optimization/real_b64_v3 --legacy-revision 9e1a380 --task1-revision c336c07 --batch-size 64 --workers 1 2 4 8 --repeats 2
python3 scripts/benchmark_collide_preparation.py --partition-source outputs/strict_optimization/real20_inputs --output-dir outputs/strict_optimization/real20_b64 --legacy-revision 9e1a380 --task1-revision c336c07 --batch-size 64 --workers 1 2 4 8 --repeats 2
python3 scripts/benchmark_collide_preparation.py --partition-source outputs/preparation_benchmark_inputs/synthetic --output-dir outputs/strict_optimization/synthetic20_b64_final --legacy-revision 9e1a380 --task1-revision c336c07 --batch-size 64 --workers 1 2 4 8 --repeats 2
```

Sono i comandi realmente eseguiti. Su un altro sistema sostituire le directory
sorgente con i propri input; su questo checkout scegliere nuove directory di
output perché quelle elencate esistono. I 20 shard reali di prova sono stati
creati dividendo ogni table da 64 righe ai confini `np.linspace(0, 64, 11,
dtype=int)` e salvando ogni slice con PyArrow, mantenendo tutte le colonne.
Nessun dato generato è necessario alla libreria o versionato.

Per 20 file su CNAF iniziare con **4 CPU worker**, batch export 64 e batch
partizioni strict 64; confrontare 256 se la RAM lo consente e 1/2/4/8 worker
sul medesimo storage/nodo. Per fast lasciare 2048 righe nella scansione di sola
provenienza. Non forzare row group interi: con oggetti nested possono essere
molto grandi; i batch configurabili e le porzioni Python da 8 MiB mantengono
il flusso limitato. Un evento singolo può superare questo budget; il batch Arrow
completo e l'inventario rimangono in RAM. Le porzioni non sono un limite di 8 MiB
alla memoria del processo.

Richiedere CPU e RAM coerenti con i worker, misurando RSS e wall sul nodo
allocato. Non aumentare automaticamente i lettori su EOS: il collo di bottiglia
può spostarsi allo storage condiviso. I risultati locali non determinano una
richiesta di RAM universale per i file completi. Usare medesimi input/seed,
versioni Python/Arrow/NumPy e filesystem per confronti; ripetere almeno tre volte
sul cluster e conservare JSON/CSV e log.

### Verifiche della modifica strict

`python3 -m pytest tests/ -q`: **200 passed, 552 warnings, 199,44 s**.
Gli avvisi sono deprecazioni Matplotlib/Pillow; log locale completo:
`outputs/strict_optimization/pytest_full.txt`. `git diff --check` passa.

I 27 nuovi casi in `tests/test_strict_fingerprints.py` mantengono un oracolo
indipendente basato su `to_pylist()` e sul JSON precedente: float16/32/64,
interi signed/unsigned fino a 64 bit, null, liste vuote, NaN/infinito, zero
negativo, slicing/chunking, struct, map, dictionary, binary, date/time/decimal e
metadata opachi. Verificano anche valori temporali non rappresentabili nascosti
sotto null, batch grandi/piccoli, schema calcolato una volta, manifest v1 salvati
con l'oracolo e worker 1/2/4/8, equivalenza di feature/label/split fast-strict
nelle due viste, invalidazione per cambio inode e modifica a stat invariati.

La suite esistente copre rilocazione, re-export, riordino e ricodifica Parquet,
corruzione con size/mtime conservati, protezioni dei manifest e gli smoke CPU
di pretraining + downstream nelle due viste e nelle due policy. Controlla anche
resume, esclusione del test dal training e rifiuto del trasferimento quando si
sostituisce il manifest fast con quello strict, pur con le stesse appartenenze.
Nessun encoder, masking, optimizer, tokenizer o formato checkpoint modificato.

È stata inoltre confrontata la mappa reale `event_id → (content_sha256, split,
label)` prima/dopo la riscrittura dei due shard in venti: identica per tutti i
128 eventi. I benchmark contro i due checkout storici confrontano l'intero SHA
del manifest per ogni variante, worker e ripetizione. I dati sorgente, i
checkpoint e gli output preesistenti sono rimasti intatti.
