# FlowyForge / M-Fond

FlowyForge mantiene una sola pipeline di classificazione: **oggetti COLLIDE →
ObjectTokenizer → Perceiver → embedding dell'evento → classification head**.
Lo stesso runner legge piccoli campioni e dataset completi in streaming, allena
su CPU/CUDA/MPS e produce valutazione e report. Il riferimento supervisionato usa
etichette di processo. Un secondo entrypoint esegue il pretraining con masking
dello stesso encoder ed esporta i pesi trasferibili; le prestazioni scientifiche
del pretraining e del transfer non sono ancora validate.

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
`--output-dir`. Il runner supervisionato non implementa il resume del training.

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
batch accoppiati o un confronto scientifico fine-tuning/linear probe.

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

La ripresa ammette solo variazioni di orizzonte epoche e device; non permette
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

Il runner supervisionato continua a inizializzare da zero per preservare il
riferimento del passo 3. Non è stata aggiunta un'opzione che trasformi di nascosto
quel riferimento in fine-tuning; qui sono disponibili export, caricamento e test
di forward con una testa nuova.

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
`preflight.json`, `checkpoint.pt`, `history.json`, `metrics.json`, `report.html`,
`tables/` e `plots/` (CSV, PNG/PDF). Aprire `report.html` con un browser; il report
non richiede un server. Conservare i file del run insieme e mantenere accessibili
i Parquet per `evaluate`. I run con partizioni condivise salvano anche
`partitions.json` e permettono la rilocazione verificata tramite `--data-root`;
i manifest legacy versione 1 conservano i percorsi originali.

Il checkpoint supervisionato è per inferenza/valutazione, senza resume; il
pretraining ha un checkpoint riprendibile separato, come descritto sopra. Training su
un singolo device, niente DDP o pesi fisici; split per riga, senza deduplicazione
di eventi fisici tra file. Accuracy/F1 su campioni piccoli non sono un benchmark.
Schema, unità e prestazioni reali COLLIDE-V2 richiedono una validazione sul sito.

- [Guida dettagliata: sampling, split, unità, memoria e valutazione](docs/classification_run.md)
- [Architettura Perceiver e contratto delle collections](docs/perceiver.md)
