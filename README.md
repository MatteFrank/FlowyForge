# FlowyForge / M-Fond

FlowyForge mantiene una sola pipeline di classificazione: **oggetti COLLIDE →
ObjectTokenizer → Perceiver → embedding dell'evento → classification head**.
Lo stesso runner legge piccoli campioni e dataset completi in streaming, allena
su CPU/CUDA/MPS e produce valutazione e report. Il riferimento supervisionato usa
etichette di processo. Un secondo entrypoint esegue il pretraining con masking
dello stesso encoder ed esporta i pesi trasferibili. Il runner di classificazione
supporta anche scratch, linear probe e fine-tuning con un comparatore dedicato;
le prestazioni scientifiche del pretraining e del transfer su dati reali non
sono ancora validate.

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

## Una pipeline, con partizioni riutilizzabili opzionali

Punto d'ingresso: **`scripts/run_classification.py`**.

```text
Parquet nested → manifest/schema/unità → stream di EventBatch con masks
              → ObjectTokenizer → Perceiver → classificazione
              → validation/test → metriche, checkpoint, report
```

| Stadio | Cosa esegue |
| --- | --- |
| `--stage partitions` | Crea una volta le appartenenze degli eventi a train/validation/test, indipendenti dal reader e dalla vista |
| `--stage inspect` | Controlla gli schemi di tutti i file e campiona i valori; scrive manifest e preflight |
| `--stage train` | Ripete il preflight, riusa `--partition-manifest` oppure costruisce gli split del run; allena, seleziona la minima validation loss, valuta il test e produce il report |
| `--stage evaluate` | Ricarica il checkpoint e rivaluta validation/test sugli stessi split del manifest |
| `--stage report` | Rigenera CSV, figure e HTML da `metrics.json`, senza leggere i Parquet |
| `--stage compare` | Verifica eventi, split, protocollo e score di due run FullReco/L1T, poi confronta le metriche |
| `--stage compare-transfer` | Verifica e confronta scratch, linear probe e fine-tuning della stessa vista |

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
| Pilot reale da file HF selezionati o Parquet locali/EOS | `prepare_collide_source.py` + `configs/data/collide_pilot_*.yaml` | `configs/runs/collide_pilot.yaml` |
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
helper è 100 eventi per processo: **è un limite del piccolo export, non del
modello o del reader**. Non produce statistiche scalari né allena il
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

## Campione reale COLLIDE: pilot da 5.000 eventi per processo

Questo percorso aggiunge una preparazione riproducibile senza sostituire il piccolo
export HF, l'esempio sintetico o il training diretto sui Parquet già disponibili.
Usa `scripts/prepare_collide_source.py` con una configurazione `configs/data/`.
I processi del primo esperimento sono:

- `tt0123j_5f_ckm_LO_MLM_semiLeptonic`;
- `WJetsToLNu_13TeV-madgraphMLM-pythia8`.

La preparazione produce **10.000 righe totali**, 5.000 per processo. Ogni riga
conserva entrambe le viste FullReco/L1T e tutte le altre colonne originali, anche
quelle non usate dal modello. I dati rimangono nested Parquet, con tipi, metadati
Arrow di schema/campo e valori originali. Nessun taglio degli oggetti, conversione
di unità o trasformazione delle feature viene salvato nell'export: questi passaggi
rimangono nel reader esistente e vengono applicati durante ispezione/training.

La selezione è uniforme senza rimpiazzo **entro i file selezionati di ciascun
processo**, con seed configurabile; le righe sono salvate in ordine di file/riga.
Non è un campione uniforme dell'intero repository COLLIDE. Le due viste usano
esattamente gli stessi eventi. Se i file non contengono abbastanza righe,
la preparazione fallisce: aggiungere file espliciti o ridurre il numero richiesto.

### Configurare la sorgente HF

Modificare [configs/data/collide_pilot_hf.yaml](configs/data/collide_pilot_hf.yaml).
Il config fissa la revisione `e67bf675673b371ee2303b3db00c59ef42375162` di
[fastmachinelearning/collide-1m](https://huggingface.co/datasets/fastmachinelearning/collide-1m/tree/e67bf675673b371ee2303b3db00c59ef42375162).
Nomi, dimensioni e footer sono stati verificati il 6 novembre 2026. I file scelti
sono quelli elencati nello YAML, uno per processo:

| Processo | Suffisso del file | Righe nel footer | Byte compressi |
| --- | --- | ---: | ---: |
| tt semileptonico | `NEVENT10000-RS30000005.parquet` | 9.827 | 1.529.191.615 |
| WJets | `NEVENT10000-RS44000011.parquet` | 8.645 | 1.304.603.082 |

Il nome `NEVENT10000` non garantisce 10.000 righe: contano i footer Parquet.
La selezione richiede **2.833.794.697 byte** di originali. Il limite iniziale è
3.000.000.000 byte; prevedere anche spazio per il campione esportato e i file
temporanei del download. I file originali rimangono sotto
`data/raw/collide_pilot/REVISIONE/PROCESSO/`.

Eseguire dalla radice, nella venv configurata sopra; i comandi sono uguali sui
diversi sistemi operativi:

```text
python scripts/prepare_collide_source.py --config configs/data/collide_pilot_hf.yaml --stage plan
python scripts/prepare_collide_source.py --config configs/data/collide_pilot_hf.yaml --stage prepare
python scripts/prepare_collide_source.py --config configs/data/collide_pilot_hf.yaml --stage inspect
```

`plan` interroga soltanto i metadati HF, verifica commit, nomi esatti e dimensioni,
e stampa il piano senza scaricare Parquet o creare l'output. La revisione deve
essere un commit SHA completo: `main` e tag mobili non sono accettati. Per
cambiare selezione, consultare i file della revisione su HF e aggiornare gli
elenchi espliciti nello YAML; non sono ammessi glob HF e non vengono aggiunti
automaticamente altri file. Un file inesistente o un totale oltre il limite
blocca la preparazione prima del download dei dati.

`prepare` ripete il controllo, scarica esclusivamente i file elencati con
`hf_hub_download`, controlla dimensione e SHA256 quando pubblicato da HF, e
mantiene gli originali. Può riutilizzare un originale locale solo se verificabile
contro i metadati remoti; un file locale diverso viene protetto, non sostituito.
Non viene usato il download dell'intero snapshot/repository. Il limite riguarda
la somma delle dimensioni dei file selezionati, inclusi quelli già presenti:
non è un misuratore del traffico HTTP, dei retry o dei dati decompressi.

### Configurare Parquet locali o EOS montato

Modificare [configs/data/collide_pilot_local.yaml](configs/data/collide_pilot_local.yaml):
impostare `source.processes` con file, directory o glob locali per i due processi.
I percorsi EOS inclusi sono esempi da confermare sul sito. Una directory viene
esplorata ricorsivamente; verificare il piano per evitare produzioni mescolate.
Per selezionare pochi file usare una lista esplicita. Lo stesso file non può
essere assegnato a processi diversi.

```text
python scripts/prepare_collide_source.py --config configs/data/collide_pilot_local.yaml --stage plan
python scripts/prepare_collide_source.py --config configs/data/collide_pilot_local.yaml --stage prepare
python scripts/prepare_collide_source.py --config configs/data/collide_pilot_local.yaml --stage inspect
```

Questi comandi non richiedono rete né `datasets`/`huggingface_hub`. Leggono i file
già accessibili senza modificarli. Sono supportati filesystem locali o montati,
non URL `root://`. In questo caso `plan` legge i footer e verifica entrambi gli
schemi; `prepare` calcola anche SHA256 degli originali, quindi ne legge tutti i
byte. Su EOS tale controllo può essere costoso: scegliere prima i file da usare.
Per lavorare offline con originali HF già scaricati, usare il config locale;
per mantenere identità HF/revisione dopo un nuovo export, usare come ingresso il
campione preparato che contiene già le colonne di provenienza.

### Opzioni di preparazione e controlli

| Opzione YAML | Significato |
| --- | --- |
| `source.kind` | `local` oppure `hf_files`; seleziona solo l'ingresso |
| `source.processes` | Mapping processo → file/directory/glob locali, oppure file HF espliciti sotto la directory del processo |
| `source.dataset`, `source.revision` | Solo HF: repository e commit SHA immutabile |
| `source.download_dir` | Solo HF: originali persistenti; deve essere separato da `output_dir` |
| `source.max_download_bytes` | Solo HF: massimo totale dei file selezionati, verificato prima del download |
| `output_dir` | Directory nuova o vuota per il campione; default `data/processed/collide_pilot` |
| `selection.events_per_process` | Eventi richiesti per processo, default 5.000, minimo 3, senza limite di 100 |
| `selection.seed` | Seed non negativo della selezione entro i file scelti |
| `selection.read_batch_size` | Righe per batch Arrow, default 64; non è il batch del modello |
| `model_config` | YAML del modello, risolto rispetto al config di preparazione; da qui provengono limiti/contratto delle collections |
| `reader.units` | Unità degli input, default GeV/rad; non riscrive i dati grezzi |
| `reader.collection_limits`, `reader.eta_clip`, `reader.value_clip` | Override del contratto per la validazione; devono corrispondere al successivo config di run |

Gli altri percorsi relativi sono risolti dalla directory di lavoro: usare la
radice del repository. Per Windows sono validi percorsi come `D:/collide/...`;
racchiudere stringhe YAML con caratteri speciali tra apici. Entrambi i config di
preparazione puntano allo stesso output: sono ingressi alternativi, non passaggi
da eseguire uno dopo l'altro. Per cambiare campione scegliere un nuovo
`output_dir` e aggiornare i percorsi nel config di run.

Il codice controlla lo schema di **tutti i file selezionati per entrambe le
viste**, prima dell'export, senza inventare alias di colonne. Dopo il salvataggio
rilegge **tutti gli eventi esportati** con le stesse trasformazioni di
`streaming.py`/`object_dataset.py`, controllando provenienza e leggibilità.
`selection.read_batch_size` limita il numero di righe per batch; la RAM usata
dipende anche da dimensione degli eventi, colonne e row group Parquet.

Le unità GeV/rad sono assunzioni dove mancano metadati riconosciuti. Il report
segnala unità assunte, campi opzionali mancanti, valori non finiti e oggetti
troncati dal reader. Il troncamento riguarda la rappresentazione del modello:
gli oggetti nel Parquet esportato restano completi. Questi controlli software
non sostituiscono la verifica delle convenzioni fisiche della produzione.

### Manifest, provenienza e riutilizzo

Il risultato contiene:

```text
data/processed/collide_pilot/
  sample_manifest.json
  preparation_config.json
  validation.json
  tt0123j_5f_ckm_LO_MLM_semiLeptonic/part-00000.parquet
  WJetsToLNu_13TeV-madgraphMLM-pythia8/part-00000.parquet
```

Viene scritto uno shard per file originale che contribuisce righe, mantenendo
anche differenze di schema/metadati tra file. Il manifest registra sorgente,
commit HF quando disponibile, processi, file logici e locali, dimensioni,
conteggi originali/esportati, row group, SHA256 e metadati originali codificati
in base64. `validation.json` contiene i controlli completi FullReco e L1T.

Ogni riga contiene le colonne `__flowyforge_origin_source`,
`__flowyforge_origin_revision`, `__flowyforge_origin_file`,
`__flowyforge_origin_row`, `__flowyforge_origin_process` e
`__flowyforge_origin_sha256`. La riga originale è indicizzata da zero nell'intero
file, non nel singolo row group. Per HF il file è il percorso logico nel repo;
per ingressi locali è l'URI assoluto originale, con revisione nulla.
Questi campi sopravvivono a un nuovo salvataggio Parquet se si conservano le
colonne; la preparazione riconosce e mantiene una provenienza già presente.
Una provenienza incompleta o malformata causa errore.

Il reader espone questa identità in `event.metadata["origin"]`; i campi esistenti
`source_file`/`source_row` continuano a indicare il file fisico e la riga letti.
Le colonne di provenienza non entrano nel tokenizer. Il piccolo export HF
storico resta utilizzabile, ma non ricostruisce retroattivamente l'identità dei
file/righe originali: per quella garanzia partire dai file originali tramite
questo nuovo percorso.

Una directory di campione non vuota non viene mai sovrascritta. In caso di errore
dopo l'avvio viene conservato un manifest con `status: failed` e la diagnostica;
non usare quel campione per il training. Scegliere una nuova directory per
riprovare; gli originali HF verificati possono essere riutilizzati. Solo
`status: complete` indica che entrambe le viste hanno superato i controlli.
Lo stadio `inspect` è in sola lettura: usa il manifest salvato, verifica SHA256
degli export e rilegge entrambe le viste, senza rete o accesso agli originali.

### Ispezionare il pilot con la pipeline esistente

Il config [configs/runs/collide_pilot.yaml](configs/runs/collide_pilot.yaml) usa
tutti gli eventi preparati, senza un secondo sottocampionamento. Se si cambiano
output, nomi dei processi, unità o limiti nel config di preparazione, aggiornare
anche questo config. Per verificare il preflight del runner:

```text
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage inspect --view FullReco --output-dir outputs/perceiver/collide_pilot_fullreco
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage inspect --view L1T --output-dir outputs/perceiver/collide_pilot_l1t
```

L'ispezione della preparazione controlla ogni riga esportata; quella del runner
crea manifest/split del run e fa un audit campionato dei valori. Nessuno dei
comandi precedenti allena il modello. Quando l'utente deciderà di effettuare
il pilot supervisionato, i comandi già implementati sono:

```text
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage train --view FullReco --device auto --output-dir outputs/perceiver/collide_pilot_fullreco
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage train --view L1T --device auto --output-dir outputs/perceiver/collide_pilot_l1t
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/collide_pilot_fullreco/checkpoint.pt --device auto --output-dir outputs/perceiver/collide_pilot_fullreco_evaluation
python scripts/run_classification.py --stage report --output-dir outputs/perceiver/collide_pilot_fullreco
```

Il default è 3 epoche; `--epochs`, `--batch-size`, `--device` e `--num-workers`
mantengono il significato descritto nella guida ai run. Le viste sono due run
separati, non un nuovo training congiunto. La preparazione non avvia training.

Verifica reale eseguita il 6 novembre 2026: scaricati e verificati i due originali
della revisione indicata; esportati 5.000 eventi per processo; riletti tutti i
10.000 eventi in entrambe le viste e completati entrambi i preflight del runner.
Verificato anche un forward CPU ObjectTokenizer + Perceiver su quattro eventi
reali per vista: embedding `[4, 64]` finiti e provenienza HF mantenuta.
Il campione occupa circa 1,70 GB oltre ai 2,83 GB degli originali. Nel report
le unità risultano assunte; tutti gli eventi superano il limite di 128 PUPPIPart
del modello, e i campi Charge opzionali di Electron/MuonTight sono assenti.
Nessun training reale è stato avviato. EOS non è stato verificato sul sito;
il percorso locale è coperto anche da test sintetici senza rete.

## Partizioni condivise fra FullReco, L1T e fasi di apprendimento

La preparazione del campione e la divisione in split sono due operazioni distinte.
`sample_manifest.json` descrive l'export del passo precedente; `partitions.json`
salva **quale evento appartiene a quale split**, senza vista, limiti degli oggetti,
unità o parametri del modello. `dataset_manifest.json` associa queste appartenenze
ai file attualmente accessibili e al reader di un singolo run. Le trasformazioni
degli oggetti restano quelle già usate dal reader COLLIDE.

### Creare una volta gli split del pilot

Preparare prima il campione con uno dei comandi HF/locali descritti sopra.
Controllare `sources.small.processes` in `configs/runs/collide_pilot.yaml`:
i nomi dei processi devono corrispondere alla provenienza nelle righe esportate.
Il pilot usa `max_events_per_class: null`, quindi tutti i 5.000 eventi esportati
per processo. Poi eseguire dalla radice, nella venv:

```text
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage partitions --partition-manifest data/processed/collide_pilot/partitions.json
```

Il comando non allena e non scarica dati. Rifiuta di sovrascrivere il manifest:
ai run successivi si passa il file esistente. Per un esperimento con una diversa
selezione/seed scegliere un nuovo nome e conservare il manifest precedente.
I percorsi CLI e `partition_manifest` nello YAML sono relativi alla directory
di lavoro; `model_config` nello YAML resta relativo al file di configurazione.
È possibile decommentare `partition_manifest` nel config del pilot per evitare
di ripetere l'opzione CLI in `inspect` e `train`.

| Opzione | Creazione e riuso |
| --- | --- |
| `--partition-manifest PATH` / YAML `partition_manifest` | Destinazione nuova nello stadio `partitions`; file da caricare in `inspect`/`train` |
| `--seed` / YAML `seed` | Alla creazione determina campione e split; al riuso il seed del manifest governa le appartenenze, quello del run l'inizializzazione del modello |
| `data.validation_fraction`, `data.test_fraction` | Default 0.2 e 0.2; il resto va al training. Devono essere positive, sommare a meno di 1 e coincidere con il manifest al riuso |
| `sources.<mode>.max_events_per_class` / `--max-events-per-class` | Campionamento uniforme senza reinserimento entro i file forniti. YAML `null` o CLI `0`: tutti. Al riuso un cap esplicito diverso viene rifiutato; `null` riusa la selezione salvata |
| `--process NOME=PERCORSO`, `--mode eos --data-root ROOT`, `--process-name NOME` | Stesse sorgenti del runner esistente; richiede almeno due processi. Non è necessario preparare un nuovo export per Parquet locali/EOS già leggibili |
| `--view FullReco` / `--view L1T`, limiti e unità del reader | Si applicano a `inspect`/`train`; non partecipano alla creazione delle appartenenze |

Per processo si ordinano le identità stabili, si riusa il campionatore con seed
già presente e si applica la stessa permutazione Feistel a sei round dello split
esistente. Le appartenenze risultanti vengono scritte esplicitamente nel manifest;
il riuso non esegue né il campionatore né la permutazione. Le frazioni predefinite
sono **60% train, 20% validation, 20% test**: con 5.000 eventi si ottengono
3.000/1.000/1.000 per processo. Gli arrotondamenti mantengono almeno un evento per
split; meno di tre eventi selezionati per processo causa errore. Un numero così
piccolo consente solo verifiche software, non una valutazione scientifica.
L'eventuale cap seleziona uniformemente nel campione disponibile: il pilot HF
rimane un campione da **file selezionati**, non uniforme nell'intero COLLIDE-1m.

### Riusare lo stesso manifest per entrambe le viste

```text
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage inspect --partition-manifest data/processed/collide_pilot/partitions.json --view FullReco --output-dir outputs/perceiver/collide_shared_fullreco
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage inspect --partition-manifest data/processed/collide_pilot/partitions.json --view L1T --output-dir outputs/perceiver/collide_shared_l1t
```

Questi due comandi verificano il reader nelle due viste e non avviano training.
Le identità degli eventi train/validation/test sono identiche anche se cambiano
vista, ordinamento fisico delle righe o limiti degli oggetti. Il codice rifiuta
eventi duplicati per identità, processi diversi, righe aggiunte/rimosse e contenuti
o schemi incompatibili. Un nuovo export deve conservare **tutto l'insieme di righe
inventariato**, comprese quelle escluse da un eventuale cap.

Quando si intende allenare, gli stessi split si usano così:

```text
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage train --partition-manifest data/processed/collide_pilot/partitions.json --view FullReco --device auto --output-dir outputs/perceiver/collide_shared_fullreco
python scripts/run_classification.py --config configs/runs/collide_pilot.yaml --stage train --partition-manifest data/processed/collide_pilot/partitions.json --view L1T --device auto --output-dir outputs/perceiver/collide_shared_l1t
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/collide_shared_fullreco/checkpoint.pt --device auto --output-dir outputs/perceiver/collide_shared_fullreco_evaluation
```

Ogni run conserva una copia `partitions.json` con checksum e un
`dataset_manifest.json` **versione 2**. Il digest del binding salvato è legato al
checkpoint, e il binding contiene il digest delle partizioni. `evaluate` usa
questa copia, non il percorso di creazione: non ricampiona e rifiuta modifiche ai
manifest. Validation seleziona il checkpoint per minima loss; il test viene
usato per la valutazione finale. I run delle due viste rimangono separati.

### Identità, esportazioni e spostamento fra macchine

Nel pilot reale controllato non sono presenti identificatori fisici verificati
di evento/run/luminosità. Il reader non tratta una colonna dal nome plausibile
come un identificatore fisico: usa `origin-v1`, un hash della sorgente logica,
revisione se disponibile, processo, file originale, riga originale a base zero
e SHA256 originale. Per HF, sorgente e file sono repo e percorso relativo alla
revisione fissata. Per dati locali/EOS senza provenienza il file originale è
identificato dal contenuto (`sha256:...`), senza mount o cache assoluti.
Non si dichiara deduplicazione fisica tra HF ed EOS o tra produzioni diverse.

Le colonne `__flowyforge_origin_*` del passo 1 preservano l'identità anche in
export successivi. `EventBatch.metadata` espone `event_id`, `origin` e la
posizione fisica attuale; le colonne di provenienza non entrano nelle feature.
Un export che perde tali colonne non può preservare l'identità dopo una
riscrittura dei file. I checksum delle righe comprendono **tutte le colonne raw,
tipi e metadati dello schema Arrow**, escludendo soltanto le colonne riservate di provenienza:
FullReco e L1T sono controllati insieme, prima delle trasformazioni del reader.

Per copiare il campione su un'altra macchina mantenere i nomi delle directory
dei processi. Per nuovi training cambiare i percorsi in `sources` o usare
`--process`, passando lo stesso `--partition-manifest`. Per rivalutare un checkpoint
su una nuova radice locale/EOS:

```text
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/collide_shared_fullreco/checkpoint.pt --data-root /nuova/radice/collide_pilot --device cpu --output-dir outputs/perceiver/collide_relocated_evaluation
```

Sostituire il percorso con quello reale, anche `C:/dati/collide_pilot` su Windows.
La radice deve contenere i processi del run. La verifica accetta file copiati
identici o riscritti/riordinati con provenienza, valori, schema Arrow e relativi metadati
conservati; nessun cambiamento delle appartenenze è consentito.

La creazione legge tutti i file raw e calcola checksum, con batch di massimo
quattro righe nella CLI. Il manifest mantiene un record per evento: spazio e
memoria **O(numero di eventi)**, anche con cap. È una scelta esplicita per il
pilot portabile. Il riuso verifica SHA256 dei file; per file identici riusa i
fingerprint delle righe, mentre gli export riscritti vengono riletti interamente.
Il reader ricontrolla il checksum di ciascun file utilizzato per iterazione/worker.
Questi controlli possono costare I/O su EOS: questo passo non introduce un indice
distribuito per l'intero dataset. Gli oggetti trasformati restano in streaming.

### Accesso per pretraining e selezione del checkpoint

L'API di accesso alle partizioni è riusata dal trainer con masking descritto
nella sezione successiva. Esempio Python eseguibile dopo la creazione:

```python
from flowyforge.training.classification_run import load_run_config
from flowyforge.data_plugins.collide_v2.streaming import (
    load_partition_manifest, bind_partition_manifest, StreamingDataset,
)

config = load_run_config("configs/runs/collide_pilot.yaml")
partitions = load_partition_manifest("data/processed/collide_pilot/partitions.json")
reader = {key: config["data"][key] for key in
          ("view", "collection_limits", "units", "eta_clip", "value_clip")}
binding = bind_partition_manifest(partitions, config["source"], **reader)
pretraining = StreamingDataset(binding, "train", purpose="pretraining")
validation = StreamingDataset(binding, "validation", purpose="checkpoint_selection")
downstream_test = StreamingDataset(binding, "test", purpose="downstream")
```

`purpose="pretraining"` richiede un binding delle partizioni condivise e ammette
**soltanto train**: validation/test causano errore. `checkpoint_selection`
ammette soltanto validation. Il test downstream resta escluso dall'ottimizzazione
e dalla selezione del checkpoint. Non sono implementati batch accoppiati
FullReco–L1T o loss fra viste.

### Compatibilità con i run precedenti

Senza `partition_manifest` tutti i comandi precedenti conservano il manifest del
dataset **versione 1**, con split impliciti e percorsi congelati. I checkpoint
esistenti continuano a essere valutabili; `--data-root` in valutazione richiede
invece la versione 2. Il manifest separato delle partizioni ha un proprio tipo
`flowyforge.event_partitions` e versione 1: non va confuso con quello del dataset.
Versioni sconosciute, checksum errati o inventari incompatibili causano errori
espliciti. Non avviene conversione automatica: l'ordinamento per identità può
produrre split diversi da quelli di un vecchio run, anche con lo stesso seed.

Verifica del passo 2 sul pilot reale, 6 novembre 2026: creato
`data/processed/collide_pilot/partitions.json`, con 3.000/1.000/1.000 eventi per
processo. Completati i preflight FullReco e L1T, confrontate tutte le 10.000
appartenenze e verificata la disgiunzione degli split; riletti otto eventi reali
per split e per vista, con identità e provenienza HF corrispondenti. Il riepilogo
è in `outputs/perceiver/collide_shared_fullreco/partition_validation.json`.
Nessun training reale eseguito; la rilocazione, il re-export e la valutazione di
checkpoint sono coperti da test sintetici CPU. L'accesso a EOS sul sito resta
da verificare. La suite completa ha superato 78 test; i warning osservati sono
deprecazioni Matplotlib/Pillow nella generazione dei PDF.

```text
python -m pytest tests/test_streaming_collide.py tests/test_sample_preparation.py tests/test_classification_run.py -q
python -m pytest tests/ -q
```

## Riferimento supervisionato ttbar / W+jets, da zero

Il passo 3 usa lo stesso classificatore Perceiver, con due training separati
FullReco e L1T. **Segnale:** `tt0123j_5f_ckm_LO_MLM_semiLeptonic`.
**Background:** `WJetsToLNu_13TeV-madgraphMLM-pythia8`.
Lo score è la probabilità softmax della classe segnale; il suo indice è risolto
dal nome nella `label_map` e salvato nei risultati (con questi nomi è 1).

La configurazione comune è
[configs/runs/collide_supervised.yaml](configs/runs/collide_supervised.yaml).
Si cambiano soltanto `--view` e `--output-dir` fra i due run. Un unico YAML evita
che due copie degli iperparametri divergano. Ogni output salva comunque la propria
configurazione risolta. Restano disponibili tutti i comandi e config precedenti.

Il modello è `foundation_tiny`: embedding 64, 16 latenti, 4 heads, 2 layers,
con le sei collections e i limiti esistenti: PUPPIPart 128, JetPuppiAK4 16,
Electron 8, MuonTight 8, PhotonTight 8, PUPPIMET 1. Feature, trasformazioni,
ordinamento, clipping e maschere non cambiano in questo passo. I pesi sono
inizializzati casualmente da zero; viene registrato `initial_state_sha256`
per verificare che i due run partano dagli stessi pesi. Non viene caricato un
checkpoint di pretraining. Il protocollo comune usa seed 42, 3 epoche, batch 16,
AdamW con learning rate 0.001, weight decay 0.0001 e gradient clipping 1.0.
Queste impostazioni definiscono il riferimento iniziale; non derivano da tuning
sul test. Stesso seed e configurazione non garantiscono risultati identici bit
per bit fra device o versioni di PyTorch.

### Prerequisiti e configurazione

Completare preparazione e partizioni dei passi 1 e 2. Il nuovo config legge gli
stessi Parquet in `data/processed/collide_pilot` e richiede
`data/processed/collide_pilot/partitions.json`: non ricrea gli split durante il
training. Se il manifest non esiste, crearlo una sola volta con il comando dello
stadio `partitions` sopra. Se si spostano i dati, aggiornare `sources.small.processes`
e `partition_manifest` nel nuovo YAML. Su EOS/HPC si possono usare percorsi già
montati o i mapping CLI `--process NOME=PERCORSO`, con gli stessi nomi dei processi
e lo stesso manifest per entrambe le viste.

Le opzioni aggiuntive sono nella sezione `evaluation` del config:

| Opzione | Default del riferimento ed effetto |
| --- | --- |
| `signal_process` | Nome esatto del ttbar semileptonico. Abilita metriche binarie e score persistenti; richiede due classi e partizioni condivise |
| `background_acceptance` | `0.01`: frazione di background accettato, cioè 1%, non 0.01%. Valore in `(0, 1]` |
| `min_background_events` | `10`: richiede `floor(N_background × background_acceptance) >= 10` per riportare l'efficienza alla frazione richiesta |
| `score_shard_rows` | `10000`: massimo numero di righe scalari per shard Parquet; non cambia il batch del modello |

I config precedenti, senza `evaluation.signal_process` o con valore `null`,
mantengono report, checkpoint e comportamento preesistenti, anche multiclass.
Le nuove impostazioni di valutazione sono salvate nel checkpoint; lo stadio
`evaluate` le ricarica dal run, senza sostituirle con quelle di un altro YAML.
Scegliere frazione di background, epoche e altri iperparametri **prima** di
consultare il test. Per modificare il protocollo creare un nuovo esperimento e
usare la validation; non scegliere il protocollo in base al confronto sul test.

### 1. Ispezione delle due viste

Gli stessi comandi funzionano dalla radice su macOS, Linux, Windows e cluster
dopo l'attivazione della venv. I percorsi dati devono essere leggibili dal nodo.

```text
python scripts/run_classification.py --config configs/runs/collide_supervised.yaml --stage inspect --view FullReco --output-dir outputs/perceiver/supervised_fullreco
python scripts/run_classification.py --config configs/runs/collide_supervised.yaml --stage inspect --view L1T --output-dir outputs/perceiver/supervised_l1t
```

Controllare `preflight.json`: schema, unità assunte o dichiarate, collections
troncate e campi opzionali. `inspect` non addestra né scrive score. Nel pilot
da 5.000 eventi per processo gli split attesi sono 3.000 train, 1.000 validation
e 1.000 test per processo, condivisi fra le viste.

### 2. Training FullReco e 3. training L1T

```text
python scripts/run_classification.py --config configs/runs/collide_supervised.yaml --stage train --view FullReco --device auto --output-dir outputs/perceiver/supervised_fullreco
python scripts/run_classification.py --config configs/runs/collide_supervised.yaml --stage train --view L1T --device auto --output-dir outputs/perceiver/supervised_l1t
```

I due run partono indipendentemente da zero. Il checkpoint scelto minimizza
esclusivamente la **validation loss**; l'epoca selezionata può essere diversa
nelle due viste pur usando lo stesso budget di training. Dopo la selezione si
rivalutano validation e test con i pesi scelti e si salvano score e metriche.
Accuracy, balanced accuracy, F1, loss, confusion matrix, history e report
rimangono disponibili. La ROC non modifica la selezione del checkpoint.
Una directory usata solo per `inspect` può essere riusata da `train`; un training
iniziato, completato o con score presenti richiede una nuova directory.

### 4. Valutazione indipendente dal training

Il training produce già la valutazione finale. Per ricalcolarla dai checkpoint:

```text
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/supervised_fullreco/checkpoint.pt --device auto --output-dir outputs/perceiver/supervised_fullreco_evaluation
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/supervised_l1t/checkpoint.pt --device auto --output-dir outputs/perceiver/supervised_l1t_evaluation
```

Si possono cambiare device, batch size o numero di worker di **valutazione**;
il protocollo di training originale resta registrato. Per dati spostati aggiungere
`--data-root NUOVA_RADICE` secondo la procedura del passo 2. Le directory con score
esistenti non vengono sovrascritte: per un'altra valutazione scegliere un nuovo
`--output-dir`. Per proseguire il training usare invece `--stage train --resume`
con il nuovo `last_checkpoint.pt`, come descritto sotto.

### Score, ROC-AUC ed efficienza del segnale

Per ciascuno split finale, `metrics.json` contiene `binary` con ROC-AUC,
`num_signal`, `num_background`, curva ROC e `operating_point`. I CSV e le figure
sono `tables/binary_validation.csv`, `tables/binary_test.csv`,
`tables/roc_validation.csv`, `tables/roc_test.csv` e `plots/roc_*.png/.pdf`.
`report.html` integra curve, conteggi, classe segnale e stato della stima.

La ROC usa `FPR = background_accettato / N_background` e
`TPR = segnale_accettato / N_segnale`. Gli score uguali entrano insieme alla
stessa soglia: l'ordine delle righe non modifica il risultato. L'AUC esatta
attribuisce mezzo punto a una coppia segnale/background con score identico.
L'efficienza riportata è la massima TPR ottenibile con una soglia `score >= soglia`
che rispetti `FPR <= background_acceptance`. Non si separano artificialmente i
pareggi e non si interpola l'efficienza fra soglie non realizzabili.
Per questo la frazione effettivamente accettata può essere inferiore al target;
vengono salvati soglia, conteggi accettati e frazione raggiunta. Una soglia `null`
con stato `ok` significa rifiutare tutti gli eventi.

Se manca una classe, ROC/AUC non sono definite (`status: missing_class`). Se il
budget `floor(N_background × target)` è inferiore a `min_background_events`,
l'efficienza è `null` con `status: insufficient_background_resolution`;
ROC e AUC rimangono disponibili quando entrambe le classi sono presenti.
Con 1.000 background e target 1% il budget è 10; con target 0.1% è soltanto 1.
Il limite configurato è una protezione sulla risoluzione empirica, non un
intervallo di confidenza. Zero background osservati dopo un taglio non dimostra
accettazione nulla nella popolazione. Le soglie del report descrivono la ROC
del rispettivo split: non sono soglie operative calibrate sulla validation
e trasferite al test. Il test fornisce una misura finale del protocollo fissato.

Gli score vengono scritti progressivamente, a partire dai batch del modello:

```text
RUN_DIR/
  scores/validation/manifest.json
  scores/validation/part-00000.parquet
  scores/test/manifest.json
  scores/test/part-00000.parquet
```

Ogni riga contiene `event_id`, `label`, `signal_score` e `split`. L'identità rimanda
alla provenienza originale in `partitions.json`. Il manifest degli score e i
metadati Parquet registrano classe segnale, label map, vista, digest del checkpoint
e dei manifest; ogni shard ha conteggio e SHA256. Gli score sono quelli del
checkpoint selezionato, non predizioni accumulate durante l'ottimizzazione.
Non si conservano in memoria oggetti fisici o tensori dell'intero dataset.
Il buffer di scrittura contiene al massimo `score_shard_rows` record scalari.
Il calcolo esatto della ROC e il confronto rileggono però **score, label e ID
scalari di uno split in memoria**, con ordinamento O(N log N) e memoria O(N);
non è implementato un ordinamento su disco per dataset arbitrariamente grandi.

### 5. Confronto FullReco / L1T

```text
python scripts/run_classification.py --stage compare --run-dir outputs/perceiver/supervised_fullreco_evaluation --run-dir outputs/perceiver/supervised_l1t_evaluation --output-dir outputs/perceiver/supervised_comparison
```

Si possono passare direttamente le due directory di training se non è stata
eseguita la valutazione separata. Il confronto legge gli artifact salvati e
non apre i Parquet fisici né ricarica il modello. Prima di confrontare metriche:

- verifica checksum e identità di tutte le partizioni, inclusi gli eventi train;
- controlla che gli score validation/test coprano esattamente gli eventi attesi,
  senza duplicati e con label corrette;
- richiede una vista FullReco e una L1T, stessa architettura, reader salvo vista,
  seed, protocollo di training e pesi iniziali;
- verifica la stessa definizione di segnale e punto operativo, poi ricalcola
  le metriche binarie dagli shard per controllare quelle registrate.

Differenze nei dati, nelle appartenenze, nelle label o nel protocollo causano
errore prima della produzione del confronto. Device, thread CPU e frequenza dei
log possono differire; il numero di worker del training fa parte del protocollo
perché influenza l'ordine degli eventi. `comparison.json`, `comparison.csv` e
`comparison.html` mostrano accuracy, F1, AUC, efficienza, conteggi e stato della
stima per entrambe le viste. Il confronto non sceglie automaticamente un modello
o iperparametri e rifiuta output già popolati.

Verifiche effettivamente eseguite per il passo 3: **84 test passati** con
`python -m pytest tests/ -q`, inclusi pareggi degli score, classi mancanti,
statistica insufficiente, shard limitati/protetti, rivalutazione e rifiuto di
confronti con eventi o impostazioni differenti. I warning sono deprecazioni
Matplotlib/Pillow nella generazione dei PDF.

È stato eseguito anche uno smoke CLI separato su **20 eventi sintetici per
classe**, con tutte le collections e l'architettura `foundation_tiny` del
riferimento, un'epoca CPU per vista: ispezione, due training, due rivalutazioni
e confronto. Gli output sono sotto `outputs/perceiver/supervised_smoke/`, con
config usato in `config.yaml` e confronto in `comparison/comparison.html`.
I due run hanno gli stessi pesi iniziali e 24/8/8 eventi complessivi per
train/validation/test. L'efficienza al target 1% è correttamente non stimabile
con soli quattro background nel test. I nomi ttbar/W+jets dello smoke sono
etichette di fixture sintetiche: quelle metriche non misurano prestazioni
fisiche. **Il training sul pilot reale non è stato avviato**, né sono stati
eseguiti training su GPU o EOS in questo passo.

## Pretraining con masking: encoder trasferibile, FullReco e L1T separati

Il passo 4 aggiunge **`scripts/run_pretraining.py`**, che riusa reader, partizioni,
collation, device e `FoundationEncoder` del classificatore. Non scarica dati,
non ricampiona gli eventi e non esegue il training supervisionato. I comandi
precedenti restano funzionanti. Non sono implementati allineamento fra viste,
batch accoppiati. Il benchmark downstream scratch/linear probe/fine-tuning è
descritto nella sezione successiva.

```text
partitions.json + Parquet nested → solo split train / validation
  → trasformazioni del reader + validità originale per feature
  → masking → ObjectTokenizer (una volta) → Perceiver → embedding evento
             └─ token corrotto dell'oggetto + embedding evento
                  → piccolo decoder della collection → feature ricostruite
  → Huber sui soli bersagli nascosti → checkpoint.pt + encoder.pt
```

### Preparazione, configurazione e ispezione

Completare la preparazione del pilot e la creazione delle partizioni documentate
sopra. I file attesi sono i Parquet nested dei due processi e
`data/processed/collide_pilot/partitions.json`. Gli stessi eventi e split usati dal
riferimento supervisionato vanno mantenuti: **non rigenerare le partizioni** per
una vista o per il pretraining. Il pilot HF rimane un campione da file selezionati,
non un campione uniforme dell'intero repository.

La configurazione comune **`configs/runs/collide_pretraining.yaml`** rende
eseguibili entrambe le viste: cambiare `--view` e `--output-dir` nei comandi sotto.
Ogni run salva il proprio `resolved_config.yaml`. Sono mantenuti architettura,
feature, trasformazioni e limiti di `configs/models/foundation_tiny.yaml`; i
parametri aggiunti sono gli indicatori di masking e i decoder. Default: seed
modello/ordine 42, 3 epoche, batch 16, AdamW con learning rate 0.001. Questi sono
parametri iniziali, non un protocollo scientifico già ottimizzato.

Dalla radice del repository, con venv attiva su macOS/Linux/Windows/HPC:

```text
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --stage inspect --view FullReco --output-dir outputs/perceiver/pretraining_fullreco
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --stage inspect --view L1T --output-dir outputs/perceiver/pretraining_l1t
```

`inspect` verifica il binding del manifest e gli schemi, percorre il training per
stimare le costanti di riferimento e la validation per contare i bersagli
mascherati e calcolare la loss delle costanti. Non aggiorna pesi e non esporta
checkpoint. Questa ispezione legge tutti gli eventi train/validation selezionati,
quindi non è necessariamente rapida su EOS. `train` può partire senza un precedente
`inspect` e riutilizzarne la directory finché non contiene artefatti di training.

Per file locali/EOS già leggibili configurare `sources.small.processes` con
mapping espliciti, oppure usare `--mode eos --data-root /percorso/montato`
e `--process-name NOME` ripetuto. Sono supportati anche `--process "NOME=PATH"`,
`--partition-manifest PATH` e `--max-events-per-class N`. I processi e il cap devono
essere compatibili con le partizioni esistenti; il cap non seleziona nuovamente
gli eventi. Identità e contenuti devono coincidere anche dopo una rilocazione.
Il runner non monta EOS, non usa `root://` e non invia job allo scheduler.

### Cosa viene nascosto e come si calcola la loss

| Opzione YAML `pretraining` | Default e significato |
| --- | --- |
| `probability` | `0.15`, probabilità indipendente per bersaglio idoneo, nell'intervallo `(0,1]` |
| `seed` | `12345`, seed delle maschere, distinto dal `seed` principale |
| `targets` | `log1p_PT` in PUPPIPart, JetPuppiAK4, Electron, MuonTight, PhotonTight; anche `log1p_Mass` in JetPuppiAK4 |
| `huber_delta` | `1.0`, soglia della Huber nelle unità delle feature trasformate |
| `decoder_hidden_dim` | `64`, larghezza del decoder `Linear → GELU → Linear` per collection |
| `collection_weights` | `{}`, peso 1 per collection; chiavi come `JetPuppiAK4` |
| `feature_weights` | `{}`, peso 1 per feature; chiavi come `JetPuppiAK4/log1p_Mass` |

`targets: null` seleziona automaticamente PT e massa jet nelle collections
abilitate; il file distribuito li elenca esplicitamente. Si può restringere
l'elenco, ma questo passo rifiuta nuovi tipi di bersaglio. Se si disabilita una
collection nel modello, rimuoverla anche da `targets` oppure usare `null`.
PUPPIMET e le feature non elencate rimangono visibili. Non vengono aggiunte
collections o nuove feature fisiche.

I pesi devono essere positivi e finiti. Esempio:

```yaml
pretraining:
  probability: 0.15
  seed: 12345
  huber_delta: 1.0
  decoder_hidden_dim: 64
  targets: null
  collection_weights: {PUPPIPart: 1.0, JetPuppiAK4: 2.0}
  feature_weights: {JetPuppiAK4/log1p_Mass: 2.0}
```

Il reader conserva `feature_valid[B,N,F]`: un bersaglio è idoneo solo se presente,
finito e, per PT/Mass, non negativo prima delle trasformazioni. Valori opzionali
mancanti, nulli, NaN/inf e valori negativi portati a zero sono esclusi. Un vero
zero è valido. I valori superiori a `value_clip` restano bersagli validi dopo il
clipping già previsto dal reader. Ordinamento, troncamento e padding preservano
la validità e gli indici originali degli oggetti.

La preparazione crea un nuovo batch senza modificare l'originale. Il valore
nascosto diventa zero e `feature_mask[B,N,F]` attiva un embedding apprendibile
specifico per collection/feature. È distinto da un vero zero; il padding ha
maschera oggetto falsa, indicatori falsi e token zero. Gli indicatori passano
attraverso collation e trasferimento al device. Il decoder riceve soltanto il
token corrotto e l'embedding globale, mai i valori originali dei bersagli.
Le etichette di processo non entrano nella loss.

La Huber viene mediata prima sugli oggetti mascherati di ciascuna feature, poi
fra le feature della collection, infine fra le collections, normalizzando i pesi
in ciascuno dei due livelli. Le feature senza bersagli nel batch sono escluse
dalle medie. Gli eventi vuoti non contribuiscono alla loss; un batch senza
bersagli nascosti salta backward e optimizer, incluso il weight decay. Un'intera
epoca senza bersagli, oppure una validation senza bersagli, causa un errore
esplicito: aumentare il campione o la probabilità. Non viene forzato un bersaglio
per ogni evento.

Le maschere di training dipendono da identità evento, seed, collection ed epoca.
Quelle di validation usano un dominio di seed distinto e rimangono fisse:
ordine dei batch, padding ed epoca non cambiano la maschera di un evento. La
valutazione aggrega somme e conteggi per feature sull'intero split, evitando di
pesare diversamente i batch incompleti.

### Eseguire i due training e leggere il confronto con la costante

I seguenti comandi avviano il pilot **solo quando eseguiti dall'utente**:

```text
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --stage train --view FullReco --output-dir outputs/perceiver/pretraining_fullreco
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --stage train --view L1T --output-dir outputs/perceiver/pretraining_l1t
```

Per CPU aggiungere `--device cpu`; per un nodo CUDA già allocato `--device cuda:0`.
`--epochs`, `--batch-size`, `--num-workers`, `--seed` e `--model-config` sovrascrivono
le opzioni corrispondenti. Learning rate, clipping, weight decay, thread CPU,
limiti degli oggetti e opzioni del masking si configurano nello YAML. Iniziare
con `--num-workers 0`; gli oggetti sono letti e ricostruiti un batch alla volta.

Il training usa esclusivamente `purpose="pretraining"` sullo split train; la
validation usa `purpose="checkpoint_selection"`. Non viene costruito un loader
per il test, né calcolata una sua loss. Le verifiche di integrità leggono i file
sorgenti e i manifest possono contenere le appartenenze test: ciò non usa quegli
eventi per ottimizzazione, costanti o selezione del checkpoint.

`metrics.json` e `history.json` riportano Huber, conteggi mascherati per feature,
eventi vuoti, batch saltati e passi dell'optimizer. La minima loss di validation
seleziona l'encoder. `validation.training_constant` valuta, sugli **stessi bersagli
mascherati**, una predizione costante per feature: la media aritmetica delle
feature trasformate valide stimata **soltanto sul train**, prima del training.
`constant_baseline` salva stime e conteggi; non si conservano gli oggetti di tutti
gli eventi in memoria. Se manca una stima train per un bersaglio incontrato in
validation, il run segnala l'incompatibilità invece di stimarla dalla validation.

`improvement_over_constant = loss_costante - loss_modello`: positivo significa
loss inferiore alla costante. La media è un riferimento semplice, non il miglior
estimatore costante possibile per una Huber. Un piccolo smoke può restare peggiore
della costante: non è un risultato scientifico di transfer. Non viene valutato
il test downstream e non viene generato un report ROC per questo task.

### Checkpoint, ripresa e caricamento nel classificatore

Ogni directory contiene:

- `checkpoint.pt`: stato dell'ultima epoca completata, encoder e decoder,
  optimizer, RNG, generatori dei loader, epoca, history, migliore encoder,
  **migliore modello completo** (encoder + decoder della stessa epoca),
  configurazione, contratto e fingerprint dei manifest;
- `encoder.pt`: **solo tokenizer e Perceiver della migliore epoca di validation**,
  insieme a contratto e provenienza; nessun decoder o classification head;
- `resolved_config.yaml`, `partitions.json`, `dataset_manifest.json`,
  `preflight.json`, `history.json`, `metrics.json`.

I due checkpoint trasferibili sono
`outputs/perceiver/pretraining_fullreco/encoder.pt` e
`outputs/perceiver/pretraining_l1t/encoder.pt`: sono encoder distinti, ciascuno
addestrato su una vista. I checkpoint hanno tipo e versione espliciti (`1`).
Registrano architettura, ordine/significato delle feature, trasformazioni,
collections, limiti, unità, vista e riferimenti/hash dei manifest.

Una directory già allenata è protetta da sovrascritture. Per riprendere
**dall'ultima epoca completata**, specificare il suo `checkpoint.pt` e la stessa
directory/configurazione. `--epochs` è il numero totale, non le epoche aggiuntive:

```text
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --view FullReco --output-dir outputs/perceiver/pretraining_fullreco --resume outputs/perceiver/pretraining_fullreco/checkpoint.pt --epochs 6
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --view L1T --output-dir outputs/perceiver/pretraining_l1t --resume outputs/perceiver/pretraining_l1t/checkpoint.pt --epochs 6
```

La ripresa ammette variazioni di orizzonte epoche, device e opzioni diagnostiche
`reporting` (che non entrano nel protocollo di ottimizzazione); non permette
cambi di batch size, maschere, reader, partizioni o dati. Non riprende a metà
batch/epoca. La riproducibilità esatta è verificata su CPU nello stesso ambiente;
cambiare hardware o versioni delle librerie non garantisce gli stessi bit.
La rilocazione dei dati può essere verificata per un nuovo binding/run, ma la
ripresa richiede lo stesso binding salvato. L'export `encoder.pt` è caricabile
indipendentemente dalla posizione dei Parquet.

Esempio Python di trasferimento con testa nuova, senza avviare fine-tuning:

```python
from flowyforge.model_plugins.foundation_encoder import (
    EventClassificationModel, encoder_contract, load_encoder,
)
from flowyforge.training.pretraining_run import load_pretraining_config

config = load_pretraining_config("configs/runs/collide_pretraining.yaml", {"view": "FullReco"})
expected = encoder_contract(config["model"], config["data"])
encoder = load_encoder("outputs/perceiver/pretraining_fullreco/encoder.pt", expected_contract=expected).eval()
classifier = EventClassificationModel.from_encoder_checkpoint(
    "outputs/perceiver/pretraining_fullreco/encoder.pt", num_classes=2,
    expected_contract=expected,
).eval()
# Con un EventBatch prodotto dal reader configurato allo stesso modo:
# embedding = encoder.encode(batch)
# logits = classifier(batch)
```

`load_encoder` ricostruisce l'architettura dal contratto e usa `strict=True` per
chiavi e dimensioni dei pesi. Passare `expected_contract` controlla anche il
contratto desiderato dal downstream, inclusa la vista: non basta che le dimensioni
coincidano. `forward_features(batch)` restituisce embedding, token, intervalli
per collection e metadati evento/oggetto, tokenizzando una sola volta. Gli indici
`object_indices` nei metadati risalgono alle posizioni originali prima di sort e
troncamento. `encode(batch)` conserva l'interfaccia precedente.

La configurazione del riferimento supervisionato del passo 3 continua a
inizializzare da zero. Il workflow downstream descritto sotto attiva esplicitamente
il caricamento dell'encoder, senza modificare il riferimento precedente.

### Report scientifico del masked pretraining

Ogni esecuzione completata di `run_pretraining.py --stage train`, incluso il
resume, genera automaticamente **`RUN_DIR/pretraining_report/pretraining_report.html`**.
Il modulo riutilizzabile è `flowyforge.analysis.pretraining_report`; non cambia
loss, maschere di training, optimizer, selezione o formato di `encoder.pt`.
Il report viene prodotto dopo il training, su CPU, con un modello separato in
`eval()` e `inference_mode()`, preservando RNG e numero di thread del chiamante.
`--stage inspect` continua a produrre solo il preflight: non ha un modello
allenato da diagnosticare. Un errore di reporting segnala un warning senza
invalidare il training già salvato; la CLI seguente consente di riprovare.

Per generare un report dagli output esistenti, dalla radice con la venv attiva
su Linux/macOS/Windows/HPC (nessun browser o display necessario sul nodo):

```text
python scripts/report_pretraining.py --run-dir outputs/perceiver/pretraining_fullreco --max-events 512
python scripts/report_pretraining.py --run-dir outputs/perceiver/pretraining_l1t --max-events 512
```

Se il report automatico esiste già, aggiungere **`--overwrite`**, oppure scegliere
una nuova directory, ad esempio `--output-dir outputs/perceiver/diagnostics_fullreco`.
La sovrascrittura è ammessa solo in una directory identificata come report;
non elimina checkpoint, dati o altri output. L'aggiornamento automatico al termine
del resume sostituisce i file del report. I file eventualmente rimasti da un
report precedente non sono elencati nell'HTML corrente: usare il suo indice.
Aprire l'HTML localmente, oppure copiare **l'intera cartella** dal cluster per
conservare figure e collegamenti.

Le impostazioni di default sono nella sezione `reporting` dello YAML comune a
FullReco/L1T. Le opzioni CLI sovrascrivono quelle salvate nel run **soltanto per
quel report**, senza riscrivere la configurazione di training:

| YAML `reporting` | CLI report | Default e significato |
| --- | --- | --- |
| `inference` | `--history-only` la disabilita | `true`; `false` genera solo curve e conteggi storici, senza leggere checkpoint o dati |
| `max_events` | `--max-events N` | `512`; massimo numero di eventi validation, non di oggetti/target; intero positivo |
| `batch_size` | `--batch-size N` | `16`; batch di inferenza CPU, worker 0, indipendente dal batch di training |
| `seed` | `--seed N` | `2718`; seed della selezione del sottocampione, distinto dal seed delle maschere salvato nel task |
| `bins` | `--bins N` | `24`; almeno 2, per istogrammi e curve di bias/risoluzione |
| `min_bin_count` | `--min-bin-count N` | `20`; almeno 2 target validi per stimare bias e deviazione standard in un bin |

Esempi di report più ampio e di consultazione senza sorgenti accessibili:

```text
python scripts/report_pretraining.py --run-dir outputs/perceiver/pretraining_fullreco --max-events 2000 --batch-size 8 --bins 20 --min-bin-count 30 --output-dir outputs/perceiver/diagnostics_fullreco_2000
python scripts/report_pretraining.py --run-dir outputs/perceiver/pretraining_l1t --history-only --output-dir outputs/perceiver/diagnostics_l1t_history
```

La selezione prende i primi `max_events` ID ordinati per
`SHA256(report-v1|seed|event_id)` nello **split validation** delle partizioni
salvate. Non dipende dalla vista, dalle etichette, dall'ordine dei batch o dai
valori fisici. Include tutti gli eventi se il cap supera la dimensione dello
split; non forza un bilanciamento fra processi. Le maschere riusano esattamente
`MaskedModelingTask.prepare_batch(validation=True)` e il seed originale del
pretraining: cambiare il seed del report cambia gli eventi scelti, non la
maschera di uno stesso evento. Il loader legge solo la validation selezionata
e usa il reader originale, inclusi clipping, troncamento e validità. Non legge
eventi test per inferenza; i controlli di integrità possono leggere byte di file
che contengono anche altre partizioni. Il campione eredita la selezione dei file
del pilot: **non è un campione uniforme dell'intero COLLIDE**.

Il report contiene:

- **Loss vs epoca:** Huber train/validation, con la costante stimata sul training
  e valutata sugli stessi target della validation. La loss train usa maschere
  variabili e pesi aggiornati durante l'epoca; non è direttamente equivalente
  a una valutazione finale train in `eval()`.
- **Heatmap per collection/feature:** `(Huber_costante − Huber_modello) /
  Huber_costante` sulla validation per ogni epoca. Positivo indica miglioramento;
  costante zero, valore mancante o nessun target danno una cella non disponibile.
  Questa quantità relativa è distinta dal miglioramento assoluto in `metrics.json`.
- **Conteggi:** target mascherati per feature/epoca, eventi vuoti, batch saltati e
  passi optimizer nei CSV. Il report conta separatamente i batch del sottocampione
  senza target mascherati: dipendono dal batch size di inferenza.
- **Ricostruzione:** istogrammi 2D predicted/true e distribuzioni true/predicted/
  baseline per ogni feature con target disponibili. Tutti usano soltanto valori
  effettivamente mascherati e originariamente validi; non includono padding.
- **Bias e risoluzione vs true PT/massa jet:** media del residuo `predicted − true`
  e deviazione standard campionaria del residuo (denominatore `N−1`), sia in GeV
  sia relativa dividendo per `true > 0`, anche per la baseline. I bin in true
  sono uniformi in `log1p(true_GeV)`. Un vero zero rimane nella diagnostica
  assoluta, ma non in quella relativa. Bin vuoti o sotto `min_bin_count` hanno
  statistiche `null`/celle CSV vuote; la soglia non è un intervallo di confidenza.
- **Embedding:** spettro della covarianza centrata, media e varianza per dimensione,
  varianza totale e rango effettivo entropico. Si usa l'encoder selezionato su input
  **non mascherati**, escludendo eventi vuoti o embedding non finiti dalla covarianza.
  Il rango massimo è `min(N−1, dimensione)`; un campione piccolo non dimostra
  collasso. Con meno di due eventi idonei la stima è esplicitamente indisponibile.

I pannelli e le colonne separano `*_transformed` da `*_GeV`. I valori fisici sono
`expm1(log1p_PT/Mass)` dopo conversione e clipping del reader: non recuperano i
valori originari prima del clipping. La baseline fisica è `expm1(media log1p sul
train)`, non la media dei PT/massa fisici e neppure l'ottimo costante della Huber.
Predizioni negative non vengono corrette artificialmente; sono conteggiate.
NaN/inf e overflow dell'inversa sono salvati come null ed esclusi dai relativi
istogrammi, con conteggi espliciti. Target della stessa particella/evento possono
essere correlati: i conteggi dei bin non sono numeri di eventi indipendenti.
Le loss del sottocampione possono differire dalla validation completa registrata
nelle curve; il report non modifica la scelta del checkpoint.

```text
RUN_DIR/pretraining_report/
  pretraining_report.html          # figure, metriche, limiti, configurazione, link
  report.json                     # fingerprint, epoca, campionamento, conteggi/stati
  resolved_config.yaml
  history.csv                     # loss, target totali, batch/eventi/optimizer
  feature_history.csv             # Huber, baseline e miglioramento per feature/epoca
  sample_events.csv               # ID di tutti gli eventi inferiti, anche vuoti
  predictions.parquet             # solo target mascherati: ID, vista, split, epoca,
                                  # collection/feature, indice originale e slot oggetto,
                                  # true/predicted/baseline trasformati e in GeV
  embeddings.parquet              # ID, split, empty, embedding non mascherato
  histograms.csv
  bias_resolution.csv
  embedding_spectrum.csv
  embedding_variance.csv
  *.png / *.pdf                   # loss, heatmap, conteggi, ricostruzioni, embedding
```

Gli artefatti di inferenza non vengono creati nei report solo storici; i CSV dello
spettro richiedono almeno due embedding idonei. Le predizioni vengono scritte
per batch in Parquet e rilette in blocchi per istogrammi e statistiche; non si
accumulano gli oggetti dell'intero campione. La covarianza richiede memoria
`O(dimensione_embedding²)`. I manifest originali e i loro ID scalari restano in
memoria come nel runner. Dimensioni di batch/reader e limiti degli oggetti
controllano la memoria dei tensori. Per aumentare `max_events` non serve cambiare
codice; costo di inferenza e dimensione del Parquet aumentano con il campione.

**Compatibilità dei checkpoint:** il checkpoint v1 aggiunge i campi opzionali
`best_model_state_dict` e `best_model_epoch`, senza cambiare i campi legacy o
l'export trasferibile. Il report controlla epoca, contratto, history, fingerprint
dei manifest e uguaglianza esatta dell'encoder selezionato con `encoder.pt` se
presente. Per checkpoint precedenti può usare il modello completo finale solo
se `epoch == best_epoch` e i pesi encoder coincidono. Se la migliore epoca è
precedente e manca il suo decoder, genera un report storico con stato
`unavailable`: non inventa il decoder e non associa pesi di epoche diverse.
Il resume legacy resta supportato; il best completo viene salvato al successivo
miglioramento, oppure recuperato subito se l'ultima epoca era la migliore.
Incompatibilità o dati modificati generano un errore esplicito; per consultare
le sole curve con dati non più accessibili usare `--history-only`.

La riproducibilità riguarda identità e maschere esatte nello stesso protocollo;
le predizioni numeriche possono differire lievemente cambiando batch size o
versione di PyTorch. Nessuna di queste diagnostiche usa il test per tuning,
misura il transfer downstream o certifica prestazioni fisiche.

Verifica del report completata il **13 novembre 2026**: `python3 -m pytest tests/ -q`
ha passato **115 test**, inclusi smoke CPU con dati sintetici, selezione della
migliore epoca diversa dall'ultima, report automatico, checkpoint legacy,
predizioni riproducibili, invalidi/padding/zeri ed eventi vuoti. Rimangono warning
di deprecazione Matplotlib/Pillow nella generazione dei PDF. Sono stati generati
e controllati anche i report dei due smoke sintetici preesistenti:

```text
python scripts/report_pretraining.py --run-dir outputs/perceiver/pretraining_smoke/fullreco --max-events 8 --bins 8 --min-bin-count 2 --overwrite
python scripts/report_pretraining.py --run-dir outputs/perceiver/pretraining_smoke/l1t --max-events 8 --bins 8 --min-bin-count 2 --overwrite
```

In questi checkpoint legacy l'ultima epoca coincide con la migliore e consente
di verificare encoder/decoder. Gli HTML sono nelle rispettive sottodirectory
`pretraining_report/`. La soglia 2 è usata qui soltanto per esercitare i bin del
report su campioni minuscoli; non è una soglia scientifica consigliata.
Non è stato avviato il training del pilot reale, né sono state misurate
prestazioni fisiche o modificato il benchmark downstream in questo intervento.

### Smoke offline riproducibile

Per provare il percorso senza HF/EOS, preparare 20 eventi sintetici per processo
e un manifest separato. Queste directory devono essere nuove; in caso contrario
scegliere altri nomi:

```text
python scripts/make_tiny_parquet_sample.py --output-dir data/samples/pretraining_tiny --events-per-class 20 --files-per-class 1
python scripts/run_classification.py --config configs/runs/local.yaml --stage partitions --process background=data/samples/pretraining_tiny/background --process signal=data/samples/pretraining_tiny/signal --max-events-per-class 20 --partition-manifest data/samples/pretraining_tiny/partitions.json
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --process background=data/samples/pretraining_tiny/background --process signal=data/samples/pretraining_tiny/signal --max-events-per-class 20 --partition-manifest data/samples/pretraining_tiny/partitions.json --view FullReco --epochs 1 --batch-size 8 --device cpu --output-dir outputs/perceiver/pretraining_tiny_fullreco
python scripts/run_pretraining.py --config configs/runs/collide_pretraining.yaml --process background=data/samples/pretraining_tiny/background --process signal=data/samples/pretraining_tiny/signal --max-events-per-class 20 --partition-manifest data/samples/pretraining_tiny/partitions.json --view L1T --epochs 1 --batch-size 8 --device cpu --output-dir outputs/perceiver/pretraining_tiny_l1t
python -m pytest tests/test_masked_pretraining.py -q
python -m pytest tests/ -q
```

Verifiche effettivamente eseguite il **7 novembre 2026**: suite `pytest tests/`
con **92 test passati**, inclusi gli 8 test mirati in
`test_masked_pretraining.py`, su CPU. I test coprono validità
originale, padding, assenza di passaggio dei valori nascosti al decoder, singola
tokenizzazione, gradienti finiti verso tokenizer/Perceiver/decoder, maschere
validation riproducibili, trasferimento rigoroso e uguaglianza degli embedding
in eval. Il resume CPU coincide esattamente con un run continuo di due epoche.

Eseguiti inoltre due smoke CLI con **40 eventi sintetici**, stessi split
24 train / 8 validation / 8 test: una prima epoca per vista, più una seconda epoca
FullReco tramite resume. Il test è rimasto escluso. Gli artefatti locali sono in
`outputs/perceiver/pretraining_smoke/`: `config.yaml`, partizioni, directory
`fullreco/`, `l1t/` e `transfer_smoke.json`. Entrambi gli export producono embedding
identici a quelli selezionati in memoria e un forward finito con testa nuova.

| Smoke | Epoca selezionata | Huber validation | Costante da train |
| --- | ---: | ---: | ---: |
| FullReco | 2 | 0.834226 | 0.572626 |
| L1T | 1 | 2.139672 | 0.578392 |

Questi valori mostrano un controllo tecnico, con epoche diverse; entrambi i
modelli sono ancora peggiori della costante. Non è un confronto scientifico
tra viste o una misura di transfer. **Nessun pretraining sul pilot reale, su GPU
o su EOS è stato avviato.**

Per ripetere l'ispezione o continuare gli smoke già presenti, senza sovrascrivere
checkpoint, usare nuove directory per l'ispezione e un orizzonte più lungo per
il resume; ad esempio:

```text
python scripts/run_pretraining.py --config outputs/perceiver/pretraining_smoke/config.yaml --stage inspect --view FullReco --output-dir outputs/perceiver/pretraining_smoke/inspect_fullreco
python scripts/run_pretraining.py --config outputs/perceiver/pretraining_smoke/config.yaml --view FullReco --output-dir outputs/perceiver/pretraining_smoke/fullreco --resume outputs/perceiver/pretraining_smoke/fullreco/checkpoint.pt --epochs 3
```

## Downstream transfer: scratch, linear probe e fine-tuning

Il benchmark usa **lo stesso `scripts/run_classification.py`**, lo stesso reader,
la stessa loss cross-entropy, gli stessi split e lo stesso ciclo di training del
riferimento supervisionato. Il pretraining con masking resta separato.
`configs/runs/collide_supervised.yaml` non è modificato: senza una sezione
`downstream` o nuove opzioni CLI il runner mantiene encoder casuale e testa MLP
legacy, inclusa la lettura dei checkpoint versione 1.

Per un confronto controllato usare **`configs/runs/collide_downstream.yaml`**:
una configurazione comune per FullReco e L1T, selezionate con `--view`, e un
`resolved_config.yaml` distinto in ogni directory. Le impostazioni di modello,
feature, collections, limiti, optimizer e campionamento sono quelle del pilot.
Il task è ttbar semileptonico contro W+jets; **il segnale è
`tt0123j_5f_ckm_LO_MLM_semiLeptonic`**. Seed 42, 3 epoche, batch 16, AdamW
learning rate 0.001 sono un protocollo iniziale riproducibile, non un tuning.

| `downstream.mode` / `--downstream-mode` | Encoder | Testa | Parametri aggiornati |
| --- | --- | --- | --- |
| `scratch` | Inizializzato casualmente | `Linear(output_dim, 2)` nuova | Encoder e testa |
| `linear_probe` | Caricato da `encoder.pt`, congelato e sempre in eval | `Linear(output_dim, 2)` nuova | Solo testa |
| `finetune` | Caricato dallo stesso `encoder.pt` del probe | `Linear(output_dim, 2)` nuova | Encoder e testa |

**Tutte e tre le modalità del benchmark hanno una testa strettamente lineare**,
senza normalizzazione, attivazione o dropout nella testa. Lo stesso seed e la
stessa architettura producono gli stessi pesi iniziali della testa, verificati
anche dal comparatore. Nel linear probe `model.train()` lascia l'encoder in
`eval()` e `requires_grad=False`; il forward dell'encoder avviene senza autograd.
L'optimizer riceve solo i parametri allenabili. Non vengono memorizzati in RAM
gli embedding di tutti gli eventi.

Lo scratch del benchmark abilita la struttura di tokenizer usata dal pretraining,
compresi gli indicatori di masking, così le chiavi e le dimensioni coincidono con
le altre modalità. Non applica masking durante la classificazione. I parametri
degli indicatori sono presenti ma non partecipano al forward downstream quando
nessuna feature è nascosta; i conteggi allenabili indicano `requires_grad=True`.
Questo scratch è un riferimento nuovo con testa lineare: non sostituisce il
riferimento legacy con testa MLP e non va mescolato con esso nel comparatore.

### Prerequisiti e configurazione

Preparare il pilot e creare una sola volta
`data/processed/collide_pilot/partitions.json`, come documentato sopra.
Per probe e fine-tuning occorre completare il pretraining della vista scelta e
avere i due export:

```text
outputs/perceiver/pretraining_fullreco/encoder.pt
outputs/perceiver/pretraining_l1t/encoder.pt
```

I due export devono provenire dalle **stesse partizioni del downstream**. Il
caricamento usa `load_encoder`/`from_encoder_checkpoint` con controllo rigoroso
di chiavi, dimensioni, architettura, vista, ordine/significato delle feature,
collections, limiti, trasformazioni e unità. Verifica inoltre la provenienza
registrata: hash delle partizioni uguale, pretraining sul train, selezione sulla
validation e `test_used: false`. Non accetta un encoder L1T per un run FullReco,
né un encoder con provenance mancante o partizioni diverse.

Le nuove opzioni sono:

```yaml
downstream:
  mode: linear_probe
  encoder_checkpoint: outputs/perceiver/pretraining_fullreco/encoder.pt
```

`--downstream-mode` e `--encoder-checkpoint` prevalgono sullo YAML.
`encoder_checkpoint` è relativo alla directory di lavoro, come i dati e gli
output. In `scratch` deve essere `null`/omesso; nelle altre modalità è obbligatorio.
Il benchmark richiede `partition_manifest` ed `evaluation.signal_process` per
produrre score identificabili e tutte le metriche binarie. Le opzioni già
esistenti per sorgenti HF materializzate, Parquet locali/EOS, device, batch e
worker rimangono disponibili. Non vengono effettuati download o mount automatici.

### Ispezione e training FullReco

Dalla radice del repository, con la venv attiva, i comandi seguenti sono uguali
su macOS, Linux, Windows e HPC. Scegliere directory nuove per nuovi esperimenti;
un training già presente è protetto da sovrascritture. Per usare CPU aggiungere
`--device cpu`, per CUDA `--device cuda:0`. `inspect` non aggiorna pesi e può
precedere il training nella stessa directory.

```text
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage inspect --view FullReco --downstream-mode scratch --output-dir outputs/perceiver/downstream_fullreco_scratch
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage inspect --view FullReco --downstream-mode linear_probe --encoder-checkpoint outputs/perceiver/pretraining_fullreco/encoder.pt --output-dir outputs/perceiver/downstream_fullreco_linear_probe
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage inspect --view FullReco --downstream-mode finetune --encoder-checkpoint outputs/perceiver/pretraining_fullreco/encoder.pt --output-dir outputs/perceiver/downstream_fullreco_finetune

python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view FullReco --downstream-mode scratch --output-dir outputs/perceiver/downstream_fullreco_scratch
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view FullReco --downstream-mode linear_probe --encoder-checkpoint outputs/perceiver/pretraining_fullreco/encoder.pt --output-dir outputs/perceiver/downstream_fullreco_linear_probe
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view FullReco --downstream-mode finetune --encoder-checkpoint outputs/perceiver/pretraining_fullreco/encoder.pt --output-dir outputs/perceiver/downstream_fullreco_finetune
```

### Ispezione e training L1T

Riutilizzare le partizioni, ma scegliere l'encoder L1T e output distinti:

```text
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage inspect --view L1T --downstream-mode scratch --output-dir outputs/perceiver/downstream_l1t_scratch
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage inspect --view L1T --downstream-mode linear_probe --encoder-checkpoint outputs/perceiver/pretraining_l1t/encoder.pt --output-dir outputs/perceiver/downstream_l1t_linear_probe
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage inspect --view L1T --downstream-mode finetune --encoder-checkpoint outputs/perceiver/pretraining_l1t/encoder.pt --output-dir outputs/perceiver/downstream_l1t_finetune

python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view L1T --downstream-mode scratch --output-dir outputs/perceiver/downstream_l1t_scratch
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view L1T --downstream-mode linear_probe --encoder-checkpoint outputs/perceiver/pretraining_l1t/encoder.pt --output-dir outputs/perceiver/downstream_l1t_linear_probe
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view L1T --downstream-mode finetune --encoder-checkpoint outputs/perceiver/pretraining_l1t/encoder.pt --output-dir outputs/perceiver/downstream_l1t_finetune
```

Questi comandi avviano il pilot solo quando eseguiti dall'utente. `train` seleziona
sempre la minima validation loss; solo dopo carica il checkpoint scelto e produce
le metriche e gli score finali di validation/test. Il test non entra nella loss
di training né nella selezione. Non utilizzare i risultati del test per cambiare
epoche, learning rate, modalità, seed o checkpoint di pretraining.

### Valutazione e confronto dei tre esperimenti

`train` include già la valutazione. Per ripeterla con un batch/device diverso,
usare una nuova directory. La modalità e il contratto vengono letti dal
checkpoint: non passare `--view`, `--downstream-mode` o `--encoder-checkpoint`
a `evaluate`.

```text
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/downstream_fullreco_finetune/checkpoint.pt --device cpu --output-dir outputs/perceiver/downstream_fullreco_finetune_evaluation
python scripts/run_classification.py --stage evaluate --checkpoint outputs/perceiver/downstream_l1t_linear_probe/checkpoint.pt --device cpu --output-dir outputs/perceiver/downstream_l1t_linear_probe_evaluation

python scripts/run_classification.py --stage compare-transfer --run-dir outputs/perceiver/downstream_fullreco_scratch --run-dir outputs/perceiver/downstream_fullreco_linear_probe --run-dir outputs/perceiver/downstream_fullreco_finetune --output-dir outputs/perceiver/transfer_comparison_fullreco
python scripts/run_classification.py --stage compare-transfer --run-dir outputs/perceiver/downstream_l1t_scratch --run-dir outputs/perceiver/downstream_l1t_linear_probe --run-dir outputs/perceiver/downstream_l1t_finetune --output-dir outputs/perceiver/transfer_comparison_l1t
```

`compare-transfer` richiede esattamente un run per modalità, tutti della **stessa
vista**. Accetta directory di training o di rivalutazione, purché il checkpoint
del classificatore sia ancora accessibile. Prima di scrivere output verifica:

- stessi ID evento e label in tutti gli split, binding coerente e hash delle
  partizioni identico; score validation/test coerenti con le appartenenze;
- stesso task, vista, contratto delle feature, architettura, seed, batch size,
  epoche, optimizer e relativo protocollo; stessa definizione di segnale e
  frazione di background accettato;
- testa lineare comune con identico fingerprint iniziale; stesso checkpoint
  preaddestrato e pesi iniziali dell'encoder per probe e fine-tuning;
- hash del checkpoint classificatore e degli shard score, metadati coerenti con
  i pesi salvati, conteggi dei parametri, encoder del probe invariato e scelta
  dell'epoca tramite minima loss di validation;
- ROC-AUC/efficienza ricalcolate dagli score persistiti.

Sono ammesse le differenze intenzionali nell'inizializzazione dell'encoder,
nel freezing e nel numero di parametri allenabili. Device, thread CPU e frequenza
di logging possono differire, ma i tempi sono confrontabili solo su risorse e
condizioni di I/O equivalenti. Il comparatore non sceglie un vincitore in base
al test. `--stage compare` conserva il precedente confronto fra FullReco e L1T
con inizializzazione casuale e non sostituisce `compare-transfer`.

### Output e interpretazione

Ogni run downstream salva gli artefatti del classificatore: configurazione
risolta, `partitions.json`, `dataset_manifest.json`, preflight, history,
`checkpoint.pt`, `metrics.json`, report HTML, CSV e figure. Gli score sono in
`scores/validation/` e `scores/test/`, in shard Parquet con `event_id`, label,
score del segnale e split.

I checkpoint downstream hanno **`format_version: 2`** e contengono encoder e
testa completi. La valutazione non ha bisogno dell'originale `encoder.pt`, pur
conservandone percorso, hash e provenienza. I checkpoint legacy versione 1 restano
caricabili per valutazione. Il resume supervisionato/downstream usa il nuovo
`last_checkpoint.pt`; il resume del pretraining mantiene la propria interfaccia.

In `metrics.json`, la sezione `downstream` registra modalità, freezing, origine
dell'encoder, fingerprint iniziali/selezionati e parametri totali/allenabili di
encoder e testa. `input_contract`, `training_protocol`, `checkpoint_sha256`,
`best_epoch` e `best_validation_loss` rendono esplicite le condizioni del run.
`performance` riporta durata aggregata del train, eventi processati su tutte le
epoche e throughput; ciascuno split valutato registra durata ed eventi/secondo.
Il tempo train include lettura/trasferimento dei batch e ottimizzazione, esclude
preflight, valutazione finale e report; il costo del pretraining non è incluso.

Il confronto salva `comparison.json`, `comparison.csv`, `comparison.html`,
`learning_curves.csv` e `plots/{roc_validation,roc_test,learning_curves}.{png,pdf}`.
L'HTML contiene le figure incorporate e si apre senza server. La tabella affianca
validation loss, best epoch, accuracy, macro-F1, ROC-AUC, efficienza del segnale,
conteggi, parametri e tempi.

Accuracy e macro-F1 usano la classe predetta; ROC-AUC misura l'ordinamento degli
score. L'efficienza è un punto della ROC empirica a FPR non superiore al target,
con score uguali accettati insieme. `insufficient_background_resolution` significa
**efficienza non stimabile**, non zero. Il target default è 1%, con almeno 10
background attesi al target: è un controllo di risoluzione, non un intervallo
di confidenza né una soglia di deployment calibrata sulla validation.

Un probe misura quanto la rappresentazione sia accessibile a una separazione
lineare; il fine-tuning misura l'adattamento supervisionato dell'intero encoder.
Il confronto a protocollo fisso non dimostra che ciascuna modalità sia ottimizzata.
Servono più seed, incertezze, controllo della rappresentatività del campione e
una procedura di tuning fissata sulla validation per conclusioni scientifiche.
Non sono introdotti pesi di luminosità/cross-section, deduplicazione fisica fra
sorgenti, Graph, MoE o allineamento fra viste.

### Riprendere il training di classificazione con `--resume`

Lo stesso runner supporta ora **legacy supervised (testa MLP), scratch, linear
probe e fine-tuning**. `--resume PATH` è ammesso solo con `--stage train`, nella
stessa directory di output. Non ricrea gli split e non reinizializza i pesi
allenati. `--epochs N` indica il **totale**: salvata l'epoca 8, `--epochs 20`
esegue le epoche **9–20**. Non è possibile ridurre l'orizzonte già configurato.

| Checkpoint di classificazione | Utilizzo |
| --- | --- |
| `checkpoint.pt` | Miglior modello secondo la minima loss di validation; encoder e testa completi. Rimane `format_version=1` per legacy, `2` per downstream. Usato da `evaluate`, `compare` e `compare-transfer` |
| `last_checkpoint.pt` | Stato dell'ultima epoca completata: pesi correnti, AdamW, RNG, generatori DataLoader, history, migliore epoca/loss, copia coerente del best, provenienza, contratto, configurazione e fingerprint. Tipo `flowyforge.classification_resume`, versione 1 |

I nuovi training salvano entrambi automaticamente. **I run prodotti dal vecchio
runner, con solo `checkpoint.pt`, non possono riprendere esattamente:** mancano
optimizer e RNG. Restano valutabili; non viene inventato uno stato di optimizer
né presentato un caricamento dei soli pesi come resume.

Esempi dalla radice con venv attiva su macOS/Linux/Windows/HPC. Presuppongono
training avviati con questa versione, stesse impostazioni YAML/CLI e stesso
ambiente. Per un linear probe FullReco già arrivato, ad esempio, a 8 epoche:

```text
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view FullReco --downstream-mode linear_probe --encoder-checkpoint outputs/perceiver/pretraining_fullreco/encoder.pt --resume outputs/perceiver/downstream_fullreco_linear_probe/last_checkpoint.pt --epochs 20 --output-dir outputs/perceiver/downstream_fullreco_linear_probe
```

Per un fine-tuning L1T:

```text
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view L1T --downstream-mode finetune --encoder-checkpoint outputs/perceiver/pretraining_l1t/encoder.pt --resume outputs/perceiver/downstream_l1t_finetune/last_checkpoint.pt --epochs 20 --output-dir outputs/perceiver/downstream_l1t_finetune
```

Per scratch e riferimento legacy, mantenendo le rispettive teste:

```text
python scripts/run_classification.py --config configs/runs/collide_downstream.yaml --stage train --view FullReco --downstream-mode scratch --resume outputs/perceiver/downstream_fullreco_scratch/last_checkpoint.pt --epochs 20 --output-dir outputs/perceiver/downstream_fullreco_scratch
python scripts/run_classification.py --config configs/runs/collide_supervised.yaml --stage train --view L1T --resume outputs/perceiver/supervised_l1t/last_checkpoint.pt --epochs 20 --output-dir outputs/perceiver/supervised_l1t
```

Per le altre combinazioni scegliere la vista e la directory del run originale:
non si può riprendere FullReco come L1T. Ripetere gli eventuali override originali
(`--device cpu`, `--batch-size`, sorgenti, modello ecc.). La CLI accetta anche
`--resume RUN_DIR/checkpoint.pt` **come alias del fratello `last_checkpoint.pt`**:
se quest'ultimo manca, restituisce un errore. `--checkpoint` rimane invece
riservato alla valutazione.

Il resume verifica prima di modificare gli output: modalità/testa/freezing,
contratto delle feature e architettura, etichette, inventario e SHA256 dei dati,
partizioni correnti e salvate, protocollo, integrità dello stato, history,
optimizer e RNG. È consentito cambiare solo il totale di epoche fra i parametri
del run; anche batch size, worker, seed, device, shuffle buffer e thread devono
rimanere invariati. Sono controllati device effettivo e versioni Python/NumPy/
PyTorch/piattaforma. Il resume richiede gli stessi percorsi/binding dei dati;
la rilocazione per `evaluate --data-root` rimane separata. Se una verifica fallisce
gli artefatti precedenti non vengono sostituiti.

Nel **linear probe**, l'encoder ripristinato rimane congelato e in eval anche
durante `model.train()`, e AdamW contiene solo i parametri allenabili della testa.
Il fingerprint dell'encoder è verificato dopo la ripresa e a ogni epoca.
Nel **fine-tuning**, i pesi ripartono dallo stato downstream dell'ultima epoca;
non vengono ricaricati dall'encoder di pretraining. La provenance originale
rimane registrata. Il file `encoder.pt` originale non è necessario per il
resume; se è ancora presente nello stesso percorso, il suo hash deve coincidere.

**Interruzioni:** si riparte dall'epoca successiva all'ultima completamente
salvata, dopo train e validation. Il lavoro di un'epoca interrotta viene rieseguito
per intero; non si salva la posizione a metà batch. Il salvataggio atomico di
`last_checkpoint.pt` precede la pubblicazione di best e history: la copia del best
al suo interno permette di recuperare un `checkpoint.pt` mancante o rimasto
all'epoca precedente. Checkpoint incompatibili/corrotti sono rifiutati.
Se l'interruzione è avvenuta nella valutazione finale, si può ripetere il comando
con lo stesso `--epochs` già raggiunto: vengono rigenerate soltanto valutazione
e presentazione, senza aggiornamenti di optimizer. Se nessuna epoca è stata
completata, non esiste ancora uno stato riprendibile.

Prima della ripresa gli score/report precedenti, anche parziali, sono conservati
in `RUN_DIR/resume_archive/run-*/`, insieme a copie dei checkpoint e manifest:
non vengono cancellati dataset né risultati precedenti. Al termine il runner
ricrea `scores/validation/`, `scores/test/`, `metrics.json` e `report.html` nella
struttura abituale, valutando **il best**, che può essere precedente al resume.
Lo stato in `last_checkpoint.pt` resta quello finale, anche quando è peggiore.
Le copie di modello/optimizer e gli archivi richiedono ulteriore spazio su disco.

`training_run` nei checkpoint, nelle metriche e nel report registra training
originale/resumed, epoca iniziale/finale dell'invocazione, totale completato,
epoche eseguite nell'invocazione, sorgente e SHA256 del resume, elenco delle
sessioni e limiti di riproducibilità. History e durata train aggregata continuano
invece dalla prima epoca; i tempi non comprendono il lavoro perso di un'epoca
interrotta. Il best mantiene i suoi pesi e la sua epoca, aggiornando i metadati
del protocollo all'orizzonte esteso. Per questo `compare-transfer` può confrontare
run continui e ripresi con lo stesso numero totale di epoche e gli stessi eventi.

La selezione continua a dipendere **solo dalla validation**. Il loader test viene
costruito dopo tutte le epoche richieste; non entra nello stato usato per allenare
o scegliere il best. Estendere un run già completato non deve essere una decisione
presa in base al suo precedente risultato sul test. Fissare il budget mediante
protocollo/validation, quindi usare il test per la valutazione finale.

La riproducibilità esatta di pesi, optimizer, RNG e metriche (esclusi tempi e
throughput) è verificata su **CPU, `num_workers=0`, stesso ambiente e hardware**.
Con worker attivi i processi vengono ricreati a ogni epoca; ripristiniamo i
generatori dei loader e `dataset.set_epoch(epoch)`, non i buffer prefetched di
un'epoca interrotta. Non si promette identità bit a bit con worker multipli,
CUDA/MPS, altro hardware o versioni differenti. Nessuna modifica a reader,
feature fisiche, split o cross-entropy è necessaria per il resume.

Verifiche del resume eseguite il **13 novembre 2026**: `python3 -m pytest tests/ -q`
→ **142 test passati**. I 27 nuovi casi usano Parquet sintetici e CPU: tutte e
quattro le modalità, confronto esatto 3 epoche continue contro 1 + resume a 3,
optimizer e RNG ripristinati, freezing, interruzione a metà epoca e fra le
scritture dei checkpoint, recupero della valutazione finale, file modificati,
stati corrotti/incompleti, CLI L1T e compatibilità dei report/comparatori.
Uno smoke con `num_workers=1` verifica il riavvio dei worker e i conteggi degli
split; non costituisce una garanzia di replay bit a bit con più worker.
Nessun training sul pilot reale né prova CUDA/MPS è stato avviato. Rimangono
warning di deprecazione Matplotlib/Pillow nella produzione dei PDF.

### Verifiche downstream effettivamente eseguite

Il 12 novembre 2026 sono stati eseguiti smoke CLI **solo su CPU e dati sintetici**,
in `outputs/perceiver/downstream_transfer_smoke_20261112/`. Il campione contiene
20 eventi per processo: 24 train, 8 validation e 8 test condivisi dalle due viste.
Il modello ridotto ha embedding/output 8, 2 latenti e un solo blocco Perceiver: non
è il pilot scientifico. Sono state eseguite una epoca di pretraining per vista,
due epoche per ciascuno dei sei run downstream, entrambi i confronti e una
rivalutazione FullReco fine-tuning con batch size 5.

Lo scratch e il fine-tuning hanno 2.426 parametri allenabili; il probe ne ha 18,
tutti nella testa. I fingerprint confermano encoder invariato nei due probe e
aggiornato nei fine-tuning. Questi controlli verificano la meccanica del transfer;
le metriche del toy sample non misurano prestazioni su ttbar/W+jets reali. Nessun
training sul pilot reale, GPU o EOS è stato avviato.

Aprire i report:

- `outputs/perceiver/downstream_transfer_smoke_20261112/comparison_FullReco/comparison.html`;
- `outputs/perceiver/downstream_transfer_smoke_20261112/comparison_L1T/comparison.html`.

La directory contiene `model.yaml`, `downstream.yaml`, `pretraining.yaml`,
`partitions.json` e `commands.json` con gli argomenti CLI effettivamente eseguiti.
Non rieseguire i comandi di training sugli stessi output; scegliere directory
nuove. Per un'ispezione del probe già configurato:

```text
python scripts/run_classification.py --config outputs/perceiver/downstream_transfer_smoke_20261112/downstream.yaml --stage inspect --view FullReco --downstream-mode linear_probe --encoder-checkpoint outputs/perceiver/downstream_transfer_smoke_20261112/pretraining_FullReco/encoder.pt --output-dir outputs/perceiver/downstream_transfer_smoke_20261112/new_inspection
python -m pytest tests/test_downstream_transfer.py -q
python -m pytest tests/ -q
```

La verifica finale `python3 -m pytest tests/ -q` ha completato **109 test passati**,
con 17 casi downstream aggiunti. Sono rimasti soltanto gli avvisi di deprecazione
Matplotlib/Pillow già presenti nei report. I test verificano caricamento e
contratti incompatibili, freezing/eval, gradienti e aggiornamenti, esclusione
del test dalla selezione, score/ID, rifiuto di protocolli o encoder diversi,
checkpoint autonomi, CLI e compatibilità legacy.

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

Ogni training supervisionato salva `resolved_config.yaml`, `dataset_manifest.json`,
`preflight.json`, `checkpoint.pt`, `last_checkpoint.pt`, `history.json`, `metrics.json`, `report.html`,
`tables/` e `plots/` (CSV, PNG/PDF). Aprire `report.html` con un browser; il report
non richiede un server. Conservare i file del run insieme e mantenere accessibili
i Parquet per `evaluate`. I run con partizioni condivise salvano anche
`partitions.json` e permettono la rilocazione verificata tramite `--data-root`;
i manifest legacy versione 1 conservano i percorsi originali.

Il best `checkpoint.pt` supervisionato rimane per inferenza/valutazione;
`last_checkpoint.pt` serve al resume. Il pretraining mantiene il proprio checkpoint
riprendibile, come descritto sopra. Training su
un singolo device, niente DDP o pesi fisici; split per riga, senza deduplicazione
di eventi fisici tra file. Accuracy/F1 su campioni piccoli non sono un benchmark.
Schema, unità e prestazioni reali COLLIDE-V2 richiedono una validazione sul sito.

- [Guida dettagliata: sampling, split, unità, memoria e valutazione](docs/classification_run.md)
- [Architettura Perceiver e contratto delle collections](docs/perceiver.md)
