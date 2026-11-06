# FlowyForge / M-Fond

FlowyForge mantiene una sola pipeline di classificazione: **oggetti COLLIDE →
ObjectTokenizer → Perceiver → embedding dell'evento → classification head**.
Lo stesso runner legge piccoli campioni e dataset completi in streaming, allena
su CPU/CUDA/MPS e produce valutazione e report. Il training è supervisionato da
etichette di processo; non è ancora un foundation model preaddestrato.

La baseline B1/MLP, la vettorizzazione scalare, il preprocessing `.npy`, il vecchio
runner HF B1 e il trainer foundation smoke duplicato sono stati rimossi. I dati e
risultati già presenti sotto `data/` e `outputs/` non vengono migrati o cancellati.

## Installazione su macOS, Linux, Windows e HPC

Serve Python >= 3.10. Eseguire dalla radice del repository. Creare la venv sul
sistema di destinazione: una venv non è trasferibile tra sistemi operativi.

macOS/Linux, incluso un nodo LXPLUS/HPC con Python disponibile:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Dopo l'attivazione, gli stessi comandi su ogni sistema:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
python -m pytest -q tests/
```

Se non si usa l'attivazione, sostituire `python` con `.venv/bin/python` su
macOS/Linux o `.venv\Scripts\python.exe` su Windows. I comandi di run qui sotto
sono su una sola riga, senza continuazioni specifiche della shell. Racchiudere
percorsi con spazi e glob tra doppi apici; su Windows si possono usare percorsi
come `C:/data/processo/*.parquet`.

Su HPC caricare prima l'ambiente Python previsto dal sito e riusarlo nel job.
Per GPU installare la build PyTorch indicata dal cluster prima dei requirements;
il file non impone una versione CUDA. Non vengono installati driver o mount EOS.
Le dipendenze HF sono necessarie solo per il download; per dati locali già pronti
è sufficiente `python -m pip install -e .`. In assenza di rete utilizzare il mirror
o le wheel approvate dal sito. Nessun comando invia automaticamente job al batch
system del cluster.

## Una pipeline, quattro stadi

Punto d'ingresso: **`scripts/run_classification.py`**.

```text
Parquet nested → manifest/schema/unità → stream di EventBatch con masks
              → ObjectTokenizer → Perceiver → classificazione
              → validation/test → metriche, checkpoint, report
```

| Stadio | Cosa esegue |
| --- | --- |
| `--stage inspect` | Controlla gli schemi di tutti i file e campiona i valori; scrive manifest e preflight |
| `--stage train` | Ripete il preflight, costruisce gli split, allena, seleziona la minima validation loss, valuta il test e produce il report |
| `--stage evaluate` | Ricarica il checkpoint e rivaluta validation/test sugli stessi split del manifest |
| `--stage report` | Rigenera CSV, figure e HTML da `metrics.json`, senza leggere i Parquet |

`train` è autonomo rispetto a `inspect`, ma non scarica dati da HF. L'ispezione
separata consente di verificare classi, schema e unità prima di impegnare risorse.
**Non esistono più stadi vectorize/preprocess separati:** trasformazioni, padding
ed estrazione delle feature avvengono nel reader, con la stessa configurazione
per training e valutazione. Il modello accetta i nested originali, non i vecchi
summary Parquet o gli array B1.

| Sorgente | Preparazione | Config del run |
| --- | --- | --- |
| Esempio sintetico offline | `make_tiny_parquet_sample.py` | `configs/runs/local.yaml` |
| Piccolo campione HF raw | `prepare_collide_source.py` | `configs/runs/hf.yaml` |
| Parquet locali già pronti | Nessuna; configurare i processi | `configs/runs/classification.yaml` + `--process` |
| Dataset su filesystem EOS/HPC | Nessuna; accesso ai file già disponibile | `configs/runs/classification.yaml` + `--mode eos` |

`small` ed `eos` scelgono i default di sorgente e campionamento, non il sistema
operativo o il modello. Iperparametri di training: YAML del run. Architettura e
limiti degli oggetti: `configs/models/foundation_tiny.yaml`.

## Primo run offline, senza dataset esterni

```text
python scripts/make_tiny_parquet_sample.py
python scripts/run_classification.py --config configs/runs/local.yaml --stage inspect
python scripts/run_classification.py --config configs/runs/local.yaml --stage train --device cpu
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/local/checkpoint.pt --device cpu --output-dir outputs/perceiver/local_evaluation
python scripts/run_classification.py --stage report --output-dir outputs/perceiver/local
```

Il generatore crea 40 eventi per classe, due classi, sei collections e entrambe
le viste FullReco/L1T sotto `data/samples/perceiver_tiny`. È un esempio del
contratto dati, non una simulazione fisica. Non sovrascrive directory non vuote.
Per ripetere il training scegliere un nuovo `--output-dir`; una directory usata
soltanto da `inspect` può essere riutilizzata da `train`.

## Hugging Face: raw → Perceiver

Controllare i processi disponibili e il config di preparazione:

```text
python scripts/list_hf_collide_dirs.py --max-dirs 30
python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m_raw.yaml
python scripts/run_classification.py --config configs/runs/hf.yaml --stage inspect
python scripts/run_classification.py --config configs/runs/hf.yaml --stage train --device auto
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/hf/checkpoint.pt --device auto --output-dir outputs/perceiver/hf_evaluation
```

La preparazione legge fino a 20 righe per processo in streaming remoto e conserva
le colonne nested in `data/hf_cache/collide-1m/perceiver_raw`. Il limite di questo
helper è 100 eventi per processo. Non produce statistiche scalari né allena il
modello. Un export incompleto causa errore e lascia un manifest diagnostico.
Un export esistente è protetto; per scaricare un nuovo campione cambiare
`paths.dataset_dir` e i percorsi corrispondenti in `configs/runs/hf.yaml`.

I nomi di processo nel repository HF e lo schema reale vanno verificati: la
preparazione raw non garantisce che tutte le collections richieste siano presenti.
Il preflight controlla nomi e tipi esatti, senza alias di colonne inventati.
Il campione remoto contiene le prime righe ricevute; il runner campiona
uniformemente solo **all'interno dei file esportati**.

Per L1T aggiungere `--view L1T` a inspect/train e usare una nuova directory:

```text
python scripts/run_classification.py --config configs/runs/hf.yaml --stage train --view L1T --device auto --output-dir outputs/perceiver/hf_l1t
```

## EOS/LXPLUS o altro filesystem HPC

Su un nodo autorizzato dove i Parquet sono leggibili come file locali, controllare
il percorso e i processi. Il root seguente è il default del progetto, da confermare
sul sito. Nessun accesso EOS viene eseguito dall'installazione o dai test.

```text
python scripts/run_classification.py --mode eos --stage inspect --data-root /eos/project/f/foundational-model-dataset/samples/ --output-dir outputs/perceiver/eos_pilot
python scripts/run_classification.py --mode eos --stage train --data-root /eos/project/f/foundational-model-dataset/samples/ --max-events-per-class 1000 --epochs 3 --batch-size 16 --device auto --output-dir outputs/perceiver/eos_pilot
```

Il discovery interpreta le directory dei processi come classi: verificare
`dataset_manifest.json`. Per scegliere solo alcune classi aggiungere
`--process-name NOME` ripetuto. Se il layout è diverso, passare mapping espliciti
con `--process "classe_a=/percorso/a" --process "classe_b=/percorso/b"`.
Su HPC-bubble o un altro cluster cambiano solo i percorsi e le risorse disponibili;
non occorre codice specifico per il cluster. Non sono supportati URL `root://`.

Dopo il pilot, esempio di full run su nodo GPU:

```text
python scripts/run_classification.py --mode eos --stage train --data-root /eos/project/f/foundational-model-dataset/samples/ --max-events-per-class 0 --epochs 20 --batch-size 64 --device cuda:0 --output-dir outputs/perceiver/eos_full
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/eos_full/checkpoint.pt --device cuda:0 --output-dir outputs/perceiver/eos_full_evaluation
```

`--max-events-per-class 0` include tutti gli eventi dei processi selezionati.
Il reader costruisce batch progressivamente; non carica tutti gli eventi in RAM.
La scansione iniziale dei footer interessa comunque tutti i file selezionati.

## Device, output e limiti

`--device auto` sceglie CUDA, poi MPS, poi CPU. `--device cpu` è il riferimento
portabile; `--device cuda:0` e `--device mps` richiedono un backend disponibile.
Il supporto nel codice non equivale a una verifica su ogni hardware/cluster.
La suite locale valida il percorso CPU; la disponibilità si può controllare con:

```text
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Ogni training salva `resolved_config.yaml`, `dataset_manifest.json`,
`preflight.json`, `checkpoint.pt`, `history.json`, `metrics.json`, `report.html`,
`tables/` e `plots/` (CSV, PNG/PDF). Aprire `report.html` con un browser; il report
non richiede un server. Conservare i file del run insieme e mantenere accessibili
i Parquet originali per `evaluate`. Il manifest congela percorsi e fingerprint:
la rilocazione dei dati tra macchine non è automatica.

Il checkpoint è per inferenza/valutazione, senza resume del training. Training su
un singolo device, niente DDP o pesi fisici; split per riga, senza deduplicazione
di eventi fisici tra file. Accuracy/F1 su campioni piccoli non sono un benchmark.
Schema, unità e prestazioni reali COLLIDE-V2 richiedono una validazione sul sito.

- [Guida dettagliata: sampling, split, unità, memoria e valutazione](docs/classification_run.md)
- [Architettura Perceiver e contratto delle collections](docs/perceiver.md)
