# Benchmark preparazione COLLIDE e partizioni

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
