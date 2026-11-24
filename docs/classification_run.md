# Classificazione: piccoli Parquet e dataset EOS

Il punto di ingresso è `scripts/run_classification.py`, configurato da
`configs/runs/classification.yaml`. Usa lo stesso modello strutturato della vertical
slice nelle modalità `small` ed `eos`. Eseguire i comandi dalla radice del repository.
Servono **Parquet originali con collections annidate**. Tutti i run usano
ObjectTokenizer + Perceiver: la pipeline scalare e il vecchio trainer smoke sono
stati rimossi. I dati e gli output storici non vengono cancellati.

Il [riferimento supervisionato ttbar/W+jets](../README.md#riferimento-supervisionato-ttbar--wjets-da-zero)
usa `configs/runs/collide_supervised.yaml`: un protocollo comune, due viste e due
directory distinte. Il README contiene i comandi di ispezione, training,
valutazione e confronto, con significato delle nuove opzioni `evaluation`.

Schema/unità reali e prestazioni scientifiche su COLLIDE-V2 devono essere
confermati sul sito. I test sintetici non sostituiscono questa validazione.

## Ambiente e configurazione

Per creare la venv su macOS/Linux o Windows e per i comandi completi locali/HF,
vedere il [README](../README.md). Dopo l'attivazione i comandi `python` sono uguali
su tutti i sistemi. Su cluster caricare l'ambiente Python del sito prima di creare
la venv e usare lo stesso ambiente nel batch job. Non copiare venv fra OS.

```text
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pip check
```

I comandi multilinea sotto usano la continuazione Bash `\`; su PowerShell o
cmd eseguirli su una sola riga senza `\`. Il README fornisce già comandi su una
riga. Percorsi e glob possono essere scritti con `/` e doppi apici anche su Windows.

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
python -c "from shutil import copyfile; copyfile('configs/runs/classification.yaml', 'configs/runs/classification_personal.yaml')"
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
  --process "signal=/percorso/reale/signal/*.parquet" \
  --process "background=/percorso/reale/background/*.parquet" \
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
  --process "signal=/percorso/reale/signal/*.parquet" \
  --process "background=/percorso/reale/background/*.parquet" \
  --max-events-per-class 128 \
  --epochs 2 --batch-size 16 --device cpu \
  --output-dir outputs/classification/small
```

Servono almeno due classi e tre eventi per classe. Si può riutilizzare la directory
di `inspect`; per ripetere un training usare una directory nuova. La presenza di
`checkpoint.pt`, `last_checkpoint.pt`, `history.json` o `metrics.json` impedisce
una sovrascrittura senza `--resume` esplicito.
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

Il [contratto delle collections](perceiver.md#data-contract) descrive
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

## Partizioni portabili opzionali

Per condividere le stesse appartenenze fra viste e fasi usare
`--stage partitions --partition-manifest PATH`, quindi passare quel file a
`inspect` e `train`. La [procedura completa nel README](../README.md#partizioni-condivise-fra-fullreco-l1t-e-fasi-di-apprendimento)
descrive creazione, opzioni, identità, API per pretraining e rilocazione.
Gli esempi seguenti senza tale opzione mantengono il comportamento legacy.

Il manifest separato registra appartenenze esplicite per identità originale,
seed, algoritmo, conteggi e checksum. La creazione non dipende dalla vista o dal
reader. I run che lo usano salvano un binding `dataset_manifest.json` versione 2
e una copia `partitions.json`; il checkpoint controlla entrambi attraverso i
digest. `evaluate` può ricevere `--data-root NUOVA_RADICE`, verificando lo stesso
insieme completo di eventi, anche dopo re-export con provenienza conservata.
Le appartenenze non vengono ricampionate. Sono verificati SHA256 dei file e,
per file riscritti, fingerprint dei contenuti/schema/metadati delle righe.

Questa modalità richiede lettura raw iniziale e indici O(numero di eventi),
anche con cap: i limiti di memoria del manifest implicito descritti sotto
si riferiscono ai run senza partizioni portabili. La lettura degli oggetti
trasformati rimane in streaming. Il pretraining può accedere esclusivamente
a train con `StreamingDataset(..., purpose="pretraining")`; validation serve
alla selezione del checkpoint, test al downstream finale. Nessun trainer di
pretraining o loss fra viste viene aggiunto da questa API.

## 3. Pilot rappresentativo dei file forniti

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
indici per processo, con conteggi esatti: nel full run legacy non viene conservato un
array di indici lungo quanto il dataset. Il manifest congela l'ordine dei file,
il seed, l'algoritmo e i conteggi necessari alla riproduzione.

La memoria contiene metadati di file/row group, batch di lettura per i processi
attivi, il buffer di shuffle, il batch del modello e i parametri. Nel pilot ci
sono anche gli indici del campione limitato. Nessun elenco di tutti gli eventi
trasformati viene accumulato. Nei run legacy le metriche sono aggregate da
loss, conteggi e matrice di confusione. Con `evaluation.signal_process` si
scrivono score scalari del checkpoint scelto in shard Parquet limitati da
`evaluation.score_shard_rows`; ROC esatta e confronto caricano score, label e ID
in memoria O(N), senza oggetti fisici o tensori di tutti gli eventi.

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
AdamW e gradient clipping. I pesi del best vengono aggiornati quando migliora la loss
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
gli stessi file sorgente. Nel percorso legacy il digest del manifest e
dimensione/mtime dei file letti vengono controllati: non sono un hash completo
dei contenuti dei Parquet. Il percorso con partizioni condivise verifica anche
SHA256 e permette la rilocazione esplicita descritta sopra.
Senza `--output-dir`, l'output è `CHECKPOINT_DIR/evaluation`. La diagnostica di
questa fase copre solo validation/test. `--batch-size` e `--num-workers` possono
essere cambiati senza cambiare la selezione; possono variare l'ordine e piccoli
arrotondamenti numerici. Per il resume del training usare `--stage train --resume
RUN_DIR/last_checkpoint.pt`; `--checkpoint` rimane riservato alla valutazione.
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
| `partitions.json` (opzionale) | Appartenenze stabili condivise; copia del manifest scelto tramite `--partition-manifest` |
| `preflight.json` | Audit limitato dei valori e riepilogo dei controlli |
| `checkpoint.pt` | Parametri del modello selezionato, configurazione e digest del manifest |
| `last_checkpoint.pt` | Ultima epoca completata: pesi, AdamW, RNG/loader, history, best completo e protocollo; resume atomico |
| `resume_archive/run-*/` | Valutazioni precedenti/parziali e copie degli artefatti conservate prima della ripresa |
| `history.json` | Training e validazione per epoca, scritto progressivamente |
| `metrics.json` | Metriche aggregate, per classe e diagnostica |
| `computation_stats.md` | CLI `train`/`inspect`, anche pretraining: wall-clock, CPU user/system del processo e dei figli terminati, hardware/software; una sezione per invocazione/resume. CUDA: intervallo dello stream, inclusivo delle attese, non tempo GPU attivo. Dettagli e limiti nel README, sezione «Device, output e limiti» |
| `scores/{validation,test}/` (opzionale) | Score della classe segnale, identità originali, label e split; shard Parquet e manifest con checksum |
| `tables/binary_*.csv`, `tables/roc_*.csv`, `plots/roc_*.png/.pdf` | Metriche binarie e curve ROC, se `evaluation.signal_process` è configurato |
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
Il manifest legacy salva percorsi assoluti e fingerprint dei file. Per spostare
i dati usare un run con partizioni condivise e `--data-root` in valutazione.
La riproducibilità riguarda input/configurazione invariati; non garantisce identità
bit a bit tra device e versioni PyTorch. Un nuovo cap o inventario produce un nuovo
manifest. Il runner usa un singolo device, senza DDP; supporta il resume ai confini
delle epoche con dati/protocollo e ambiente invariati.

## Resume di classificazione

La guida operativa con esempi FullReco/L1T per legacy, scratch, linear probe e
fine-tuning è nel [README](../README.md#riprendere-il-training-di-classificazione-con---resume).
`run_classification(config, resume=path)` riusa lo stesso ciclo di training.
`--epochs` è il numero totale richiesto, mai il numero aggiuntivo; il protocollo
esclude soltanto l'orizzonte epoche e i percorsi dei file di configurazione/output.
Tutte le altre impostazioni, incluso il runtime effettivo, devono coincidere.
I checkpoint inference v1/v2 rimangono leggibili; non contengono lo stato per
ripartire esattamente e non sono convertiti in checkpoint di resume.

`training/checkpointing.py` contiene il salvataggio atomico e gli snapshot RNG
già usati dal pretraining, ora condivisi. `last_checkpoint.pt` ha tipo
`flowyforge.classification_resume`, versione 1, più un checksum strutturale
dei metadati e dei tensori. Comprende stato Python/NumPy/Torch CPU e, sul device
usato, CUDA o MPS, generatori train/validation, nomi ordinati dei parametri
AdamW, momenti, epoca, history, diagnostica della prima epoca, provenienza,
contratto, hash dei file/manifest e snapshot del best. Anche i manifest legacy
v1 hanno qui un hash separato di ogni file per il controllo del resume; non
viene modificato il loro algoritmo di campionamento né il reader.

L'ordine di salvataggio per epoca è `last_checkpoint.pt` atomico, poi
`checkpoint.pt` atomico e `history.json`. Il primo è lo stato di riferimento:
un'interruzione fra le scritture lascia un best/history recuperabile dal suo
snapshot. Uno stato finale completo con valutazione interrotta ammette resume
allo stesso orizzonte per rigenerare solo gli output finali. Se non esiste una
prima epoca completata, non è disponibile il resume.

La validazione avviene prima di archiviare/sostituire gli artefatti: contratto,
dataset/partizioni, checksum, history contigua e selezione minima validation,
chiavi/forme/pesi finiti, gruppi AdamW e forme dei momenti, RNG, provenienza e
policy di freezing. Il modello viene costruito con `initialize_encoder=False`
e riceve strict i pesi downstream, mai l'inizializzazione pretrained. Nel probe
si verifica l'encoder in eval, senza parametri allenabili e con fingerprint
immutato; l'optimizer contiene solo la testa. Il pretrained originale può
essere assente; se presente, deve mantenere l'hash registrato.

Prima di continuare si conservano score/report e copie dei checkpoint/manifest
in `resume_archive/`. I nuovi score usano il fingerprint del best aggiornato e
il percorso standard. Il test è letto solo dopo la selezione finale; i risultati
di un precedente test non vengono letti per adattare il training. Il report
espone `training_run` con sessioni, epoche eseguite, provenienza del resume e
limiti di riproducibilità. I comparatori ignorano l'interruzione come differenza
di inizializzazione, ma continuano a richiedere lo stesso orizzonte totale,
protocollo, identità, score verificati e checkpoint compatibile.

I loader non usano worker persistenti. L'epoca del dataset viene impostata sul
numero assoluto della nuova epoca, preservando mixing/shuffle e assegnazione
dei row group; i generatori ripristinati determinano i seed dei nuovi worker.
I buffer di prefetch non sono salvati: il resume è esclusivamente ai confini
delle epoche. L'uguaglianza bit a bit è testata su CPU/worker 0 nello stesso
ambiente; timing e throughput differiscono, e non si garantisce identità su
acceleratori, più worker o altro hardware. `tests/test_classification_resume.py`
include replay continuo vs interrotto, freezing, optimizer, crash a metà epoca
e fra checkpoint, rigetti prima delle scritture e compatibilità dei confronti.

## Verifiche software

I test nuovi usano solo piccoli Parquet sintetici. Il test di integrazione esegue
una singola epoca CPU su 40 eventi e una rivalutazione del checkpoint:

```bash
python -m pytest -q tests/test_streaming_collide.py tests/test_classification_run.py tests/test_streaming_report.py
```

Prima di un eventuale commit, le istruzioni del repository richiedono anche
`python -m pytest tests/`. Non aggiungere dataset, checkpoint o output pesanti al Git.

Il pretraining con masking ha un entrypoint separato, `scripts/run_pretraining.py`,
che riusa le partizioni e l'encoder senza cambiare questo riferimento supervisionato.
Il suo checkpoint completo supporta resume; `encoder.pt` supporta caricamento
rigoroso nel classificatore con testa nuova. Procedura e limitazioni nel
[README](../README.md#pretraining-con-masking-encoder-trasferibile-fullreco-e-l1t-separati).

## Estensione downstream: punti di integrazione e compatibilità

L'audit del riferimento `main@89f609e` ha identificato `_make_model`,
`_run_epoch`, il salvataggio/caricamento del checkpoint e i controlli del
comparatore come punti di estensione. Il reader e il Perceiver non sono modificati.
Il training supervisionato rimane in un solo ciclo; non esiste un secondo
framework o un runner downstream duplicato.

| Componente | Responsabilità downstream |
| --- | --- |
| `load_run_config` | Sezione opzionale `downstream`; tre modalità, checkpoint e prerequisiti del benchmark |
| `_make_model` | Legacy invariato; benchmark con testa lineare comune, caricamento tramite API esistente e contratto rigoroso |
| `EventClassificationModel.train/encode` | Encoder congelato sempre in eval e senza autograd nel probe |
| `_run_epoch` | Stesso ciclo cross-entropy; optimizer solo sui parametri allenabili, durata/throughput |
| `run_classification` | Provenienza, selezione validation, fingerprint, score e checkpoint v2 |
| `evaluate_run` | Ripristino completo da checkpoint v1/v2, senza dipendenza dal file encoder sorgente |
| `_load_comparison_run` | Controlli comuni di partizioni, identità e score per entrambi i comparatori |
| `compare_downstream_runs` | Protocollo comune, freezing e fingerprint; tabelle, ROC e learning curves |

La sezione `downstream` assente conserva la testa MLP legacy, le chiavi dei pesi
e la sequenza di inizializzazione casuale. La configurazione
`collide_supervised.yaml` resta invariata. Il benchmark esplicito usa invece
`nn.Linear(output_dim, num_classes)` in tutte le modalità, con la struttura del
tokenizer del pretraining anche in scratch. L'identità iniziale della testa viene
verificata; i pesi iniziali dell'encoder possono differire tra scratch e transfer.

I checkpoint v2 salvano anche contratto, provenienza e stato del freezing. Per
valutare si ricostruisce l'architettura e si carica l'intero state dict con controllo
rigoroso; non si riapre il file `encoder.pt` originale. Il comparatore richiede il
checkpoint classificatore, verificato tramite SHA256, ma non i dati raw né
l'export pretraining originale. Per directory di rivalutazione usa il percorso
`checkpoint_path` registrato. Mantenerlo accessibile dopo eventuali spostamenti.

I controlli della provenienza si riferiscono ai metadati e agli hash presenti
negli artefatti; non dimostrano identità fisica tra eventi di sorgenti diverse e
non sostituiscono un audit della produzione dei dati. Il protocollo non ammette
pretraining su partizioni diverse in questo benchmark.

Le durate sono wall-clock e includono l'I/O dei batch. Il throughput è il numero
di eventi processati diviso la somma delle durate train; non misura FLOPS, tempo
di solo kernel o accelerazione di hardware differenti. Le curve train riportano
metriche online, quelle validation il modello a fine epoca. Solo il checkpoint
scelto sulla validation viene utilizzato per gli score finali di test.

Comandi completi per entrambe le viste, opzioni e interpretazione nel
[README](../README.md#downstream-transfer-scratch-linear-probe-e-fine-tuning).
