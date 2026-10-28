# Classificazione: piccoli Parquet e dataset EOS

Il punto di ingresso è `scripts/run_classification.py`, configurato da
`configs/runs/classification.yaml`. Usa lo stesso modello strutturato della vertical
slice nelle modalità `small` ed `eos`. Eseguire i comandi dalla radice del repository.
Servono **Parquet originali con collections annidate**, non i riassunti scalari B1
creati da `make_tiny_parquet_sample.py`, né gli array `X.npy` della baseline MLP.
I comandi precedenti per B1 e per lo smoke test rimangono disponibili.

Non sono stati eseguiti training scientifici su EOS durante questa implementazione.
L'accesso alla directory indicata restituisce `Operation not permitted` in questo
ambiente: schema e unità reali devono essere confermati con il preflight da un nodo
CERN autorizzato. Non confondere i test sintetici con una validazione di COLLIDE-V2.

## Ambiente e configurazione

Usare un ambiente con le dipendenze del progetto. Quello presente nel workspace è:

```bash
source /eos/home-m/matteo/venvs/venv_flowyforge/bin/activate
python -m pip install -e .
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Su un'altra macchina, creare un ambiente e installare il progetto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Per CUDA servono un nodo GPU e una build PyTorch compatibile con i driver.
`--device auto` sceglie CUDA, poi MPS, poi CPU. `--device cuda:0` richiede la GPU:
se non è disponibile, il runner termina con un errore prima di leggere il dataset.
Non viene effettuata alcuna installazione automatica di dipendenze.

Nel file YAML, `sources.small` e `sources.eos` mantengono le due modalità. Il modello
e i limiti degli oggetti provengono da `configs/models/foundation_tiny.yaml`.
`model_config` è relativo al file YAML; i percorsi dei dati e degli output sono
relativi alla directory di lavoro. `--model-config` invece è relativo alla directory
di lavoro. Per conservare configurazioni personali:

```bash
cp configs/runs/classification.yaml configs/runs/classification_personal.yaml
```

Si possono salvare i percorsi piccoli sotto `sources.small.processes`, oppure
passarli con `--process`. Gli argomenti CLI prevalgono sullo YAML. Il seed predefinito
è 42; `--seed` lo modifica. Scegliere nomi di classe stabili e distinti.

## 1. Piccoli Parquet: preflight e primo run

Sostituire i percorsi segnaposto con due file, directory o glob di Parquet originali.
Ogni `--process NOME=PERCORSO` definisce una classe; racchiudere i glob tra apici.

```bash
python scripts/run_classification.py \
  --config configs/runs/classification.yaml \
  --mode small --stage inspect \
  --process 'signal=/percorso/reale/signal/*.parquet' \
  --process 'background=/percorso/reale/background/*.parquet' \
  --max-events-per-class 128 \
  --output-dir outputs/classification/small
```

`inspect` legge i footer di tutti i file selezionati e un campione uniforme di
contenuti, senza inizializzare un training. Controllare `dataset_manifest.json`
e `preflight.json` secondo la checklist più sotto. Poi:

```bash
python scripts/run_classification.py \
  --config configs/runs/classification.yaml \
  --mode small --stage train \
  --process 'signal=/percorso/reale/signal/*.parquet' \
  --process 'background=/percorso/reale/background/*.parquet' \
  --max-events-per-class 128 \
  --epochs 2 --batch-size 16 --device cpu \
  --output-dir outputs/classification/small
```

Servono almeno due classi e tre eventi per classe. Si può riutilizzare la directory
di `inspect`; per ripetere un training usare una directory nuova. La presenza di
`checkpoint.pt`, `history.json` o `metrics.json` impedisce una sovrascrittura.
Ogni training ricostruisce e verifica il manifest prima di iniziare.

## 2. EOS: controllare layout e dati senza training

```bash
python scripts/run_classification.py \
  --config configs/runs/classification.yaml \
  --mode eos --stage inspect \
  --data-root /eos/project/f/foundational-model-dataset/samples/ \
  --output-dir outputs/classification/eos_inspect
```

Il discovery interpreta le sottodirectory immediate come classi e cerca i Parquet
ricorsivamente in ciascuna. Se l'unico contenitore è `production_final`, entra in
quel contenitore. Campagne `production*` multiple o file senza classe alla radice
richiedono una scelta esplicita. **Controllare i nomi delle classi nel manifest**:
il filesystem non permette di dedurre automaticamente il significato fisico di
una directory. Per selezionare alcune classi usare `--process-name NOME` ripetuto.
Per layout diversi, fornire direttamente i mapping:

```bash
python scripts/run_classification.py \
  --mode eos --stage inspect \
  --process processo_a=/eos/percorso/reale/processo_a \
  --process processo_b=/eos/percorso/reale/processo_b \
  --output-dir outputs/classification/eos_explicit_inspect
```

Il preflight verifica lo schema di **tutti** i file selezionati; l'audit dei valori
legge al massimo `data.audit_events_per_class` righe per classe, 32 per default,
campionate sull'intero processo. La scansione dei footer può richiedere tempo
anche senza training. Le anomalie rare fuori dal campione saranno controllate
quando gli eventi vengono letti nel training o nella valutazione.

Controllare questi elementi prima di aumentare la scala:

| Controllo | Dove e come leggerlo |
| --- | --- |
| Classi e selezione | `dataset_manifest.json`: `label_map`, file, righe disponibili/selezionate e `split_counts` |
| Schema e collections | Per ogni file: `columns`, tipi, `missing_optional_columns`. I nomi richiesti sono esatti, senza alias automatici |
| Unità | `preflight.json`: `input_units`, `model_units`, `unit_metadata_fields`, `unit_assumed_fields` e warning |
| Valori mancanti/non finiti | `fields`: liste null, valori non finiti e min/max **nelle unità originali**; `collections`: sostituzioni e campi opzionali assenti |
| Troncamenti | Totali originali, conservati e scartati per collection; numero di eventi troncati |

Il [contratto della vertical slice](foundation_smoke.md#data-contract) descrive
nomi, feature, maschere e trasformazioni. Il default è `FullReco`; `--view L1T`
richiede realmente le colonne `L1T_...` corrispondenti. Ogni collection abilitata
deve avere un limite finito in `collection_limits`; zero la disabilita. Non
disabilitare una collection mancante prima di aver verificato i file e i nomi.

Lo YAML dichiara `data.units.momentum: GeV|MeV` e `angle: rad|deg`. I valori sono
convertiti in GeV/radianti **prima** di clipping e trasformazioni. Il confronto con
i metadati supporta `unit`/`units` sul campo Arrow, una mappa JSON schema `units`
indicizzata per nome esatto di colonna, oppure `momentum_unit`/`angle_unit` sullo
schema. Una contraddizione causa errore; metadati assenti o in formati diversi
sono segnalati come assunzioni. I range dei valori aiutano il controllo, ma non
certificano le unità fisiche. Non vengono inferite unità dalla distribuzione.

I campi opzionali mancanti sono riempiti con zero; NaN/inf/null numerici diventano
zero. I contatori registrano le sostituzioni. Le liste null rappresentano collection
vuote; lunghezze incoerenti tra campi o strutture annidate ambigue causano errore
con file e riga. Gli oggetti sono ordinati per PT e troncati secondo il contratto
esistente; MET conserva l'ordine originale.

## 3. Pilot rappresentativo

```bash
python scripts/run_classification.py \
  --config configs/runs/classification.yaml \
  --mode eos --stage train \
  --data-root /eos/project/f/foundational-model-dataset/samples/ \
  --max-events-per-class 10000 \
  --epochs 5 --batch-size 64 --device auto \
  --output-dir outputs/classification/eos_pilot
```

Il limite seleziona un campione uniforme senza reinserimento su **tutte le righe**
dei file di ciascuna classe. Non seleziona le prime righe e non assegna la stessa
quota a file di dimensioni diverse. Gli indici del campione limitato sono salvati
nel manifest. Le frazioni di validazione e test sono configurabili, entrambe 0.2
per default, con almeno un evento di ogni classe in ciascuno split.

Un cap uguale per classe modifica le proporzioni rispetto al dataset completo:
interpretare accuracy, precision e F1 insieme ai supporti. Non sono applicati pesi
di evento, sezioni d'urto o bilanciamento artificiale nel full run. La miscelazione
dei processi tiene conto delle righe rimanenti; il training usa anche un buffer
di shuffle limitato e cambia ordine a ogni epoca conservando gli stessi split.

Aprire `report.html` per controllare andamento delle loss, prestazioni per classe,
matrici di confusione, supporti e diagnostica. Usare la validazione per scegliere
gli iperparametri; mantenere il test per la valutazione finale.

## 4. Full run in streaming su GPU

```bash
python scripts/run_classification.py \
  --config configs/runs/classification.yaml \
  --mode eos --stage train \
  --data-root /eos/project/f/foundational-model-dataset/samples/ \
  --max-events-per-class 0 \
  --epochs 20 --batch-size 64 --device cuda:0 \
  --output-dir outputs/classification/eos_full
```

`0` sulla CLI oppure `null` nello YAML include tutti gli eventi delle classi
selezionate. Gli split usano una permutazione deterministica non lineare degli
indici per processo, con conteggi esatti: nel full run non viene conservato un
array di indici lungo quanto il dataset. Il manifest congela l'ordine dei file,
il seed, l'algoritmo e i conteggi necessari alla riproduzione.

La memoria contiene metadati di file/row group, batch di lettura per i processi
attivi, il buffer di shuffle, il batch del modello e i parametri. Nel pilot ci
sono anche gli indici del campione limitato. Nessun elenco di tutti gli eventi
trasformati o delle predizioni viene accumulato. Le metriche sono aggregate da
loss, conteggi e matrice di confusione.

Per regolare l'uso di risorse:

| Impostazione | Effetto |
| --- | --- |
| `--batch-size` | Memoria del modello/GPU; ridurlo se manca VRAM |
| `data.read_batch_size` | Righe Arrow per lettura; ridurlo se gli eventi grezzi sono grandi |
| `data.shuffle_buffer` | Numero massimo di eventi trasformati nel buffer di shuffle per worker |
| `--num-workers` | Default 0. Provare 2 dopo il pilot se la lettura limita il throughput; i worker dividono i row group senza duplicazioni |
| `training.cpu_threads` | Thread PyTorch del processo principale |
| `training.log_every_batches` | Frequenza dei log con loss, eventi elaborati e throughput |

Con worker multipli crescono memoria, file aperti e letture EOS: ogni worker ha
i propri buffer e due batch di prefetch. Il contesto multiprocessing è `spawn`,
adatto anche quando il processo principale usa CUDA. Lo shuffle è limitato in
memoria, non una permutazione globale degli eventi già caricati.

Tutti i parametri di tokenizer, encoder e classificatore vengono addestrati con
AdamW e gradient clipping. Il checkpoint viene aggiornato quando migliora la loss
di validazione. Il test viene letto dopo la selezione del modello. I contatori di
diagnostica finale coprono una volta le righe selezionate: prima epoca di training
e valutazione finale di validation/test; l'audit campionato rimane separato.

## 5. Valutazione e report separati

```bash
python scripts/run_classification.py \
  --stage evaluate \
  --checkpoint outputs/classification/eos_full/checkpoint.pt \
  --device cuda:0 \
  --output-dir outputs/classification/eos_full_evaluation
```

Vengono ricalcolate validation e test sul manifest originale, senza nuovi split.
Conservare insieme checkpoint, manifest, preflight e history e mantenere accessibili
gli stessi file sorgente. Il digest del manifest e dimensione/mtime dei file letti
vengono controllati: non sono un hash completo dei contenuti dei Parquet.
Senza `--output-dir`, l'output è `CHECKPOINT_DIR/evaluation`. La diagnostica di
questa fase copre solo validation/test. `--batch-size` e `--num-workers` possono
essere cambiati senza cambiare la selezione; possono variare l'ordine e piccoli
arrotondamenti numerici. `--checkpoint` non implementa resume del training.
Usare una directory di valutazione separata: gli output di training sono protetti
anche da sovrascritture tramite questa fase.

Per rigenerare soltanto la presentazione:

```bash
python scripts/run_classification.py \
  --stage report \
  --output-dir outputs/classification/eos_full
```

Questa fase legge solo `metrics.json`, senza training né lettura dei Parquet.

## Output e interpretazione

| File/directory | Contenuto |
| --- | --- |
| `resolved_config.yaml` | Configurazione effettiva del run |
| `dataset_manifest.json` | Classi, file, schema, unità dichiarate, campionamento e split |
| `preflight.json` | Audit limitato dei valori e riepilogo dei controlli |
| `checkpoint.pt` | Parametri del modello selezionato, configurazione e digest del manifest |
| `history.json` | Training e validazione per epoca, scritto progressivamente |
| `metrics.json` | Metriche aggregate, per classe e diagnostica |
| `report.html` | Report autonomo, con immagini incorporate, condivisibile senza server |
| `tables/*.csv` | Riepilogo, history, metriche per classe, matrici di confusione, supporti e diagnostica |
| `plots/*.png`, `plots/*.pdf` | Curve di apprendimento, metriche per classe, composizione degli split, confusioni e diagnostica |

Le matrici hanno classi vere sulle righe e predette sulle colonne; sono esportate
sia in conteggi sia normalizzate per classe vera. La balanced accuracy media la
recall delle classi con supporto; la macro F1 considera tutte le classi configurate.
I punteggi non definiti sono zero. Le curve di training aggregano predizioni durante
l'ottimizzazione, mentre i riepiloghi di validation/test usano il modello selezionato.

Le etichette sono quelle delle directory/mapping, non inferenze dalla fisica degli
eventi. Gli split sono separati per riga sorgente; **non** si deduplicano eventi
fisici replicati in file diversi. Prima di risultati scientifici definitivi,
verificare identità degli eventi, eventuali correlazioni tra campagne e pesi fisici.
La riproducibilità riguarda input/configurazione invariati; non garantisce identità
bit a bit tra device e versioni PyTorch. Un nuovo cap o inventario produce un nuovo
manifest. Il runner usa un singolo device, senza DDP né resume.

## Verifiche software

I test nuovi usano solo piccoli Parquet sintetici. Il test di integrazione esegue
una singola epoca CPU su 40 eventi e una rivalutazione del checkpoint:

```bash
python -m pytest -q tests/test_streaming_collide.py tests/test_classification_run.py tests/test_streaming_report.py
```

Prima di un eventuale commit, le istruzioni del repository richiedono anche
`python -m pytest tests/`. Non aggiungere dataset, checkpoint o output pesanti al Git.
