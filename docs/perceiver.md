# Perceiver: architettura e contratto dati

La rete attiva è `EventClassificationModel`, definita in
`src/flowyforge/model_plugins/foundation_encoder.py`. Contiene un
`FoundationEncoder` condiviso (`ObjectTokenizer` + `PerceiverEventEncoder`) e una
testa supervisionata che predice il processo. I pesi vengono inizializzati e
allenati insieme; non si carica un modello preaddestrato.

```text
EventBatch: collections eterogenee [B, N_g, F_g] + masks [B, N_g]
  -> proiezione Linear(F_g, 64) per collection + embedding del tipo
  -> token concatenati [B, N_totale, 64]
  -> cross-attention: 16 latenti apprendibili interrogano i token
  -> feed-forward + 2 blocchi di self-attention sui latenti
  -> media dei latenti + LayerNorm + Linear(64, 64)
  -> embedding evento [B, 64]
  -> LayerNorm + Linear(64, 64) + GELU + Linear(64, C)
  -> logits [B, C]
```

Le dimensioni provengono da `configs/models/foundation_tiny.yaml`: 4 attention
heads, feed-forward 64→256→64, dropout 0.1, connessioni residue. L'attenzione fra
oggetti avviene attraverso i latenti; la self-attention è sui 16 latenti. Senza
positional embeddings, il modello tratta gli oggetti come insiemi.

I file di implementazione sono `representation/tokenizer.py`,
`model_plugins/perceiver.py`, `model_plugins/foundation_encoder.py` e
`training/classification_run.py`. Il runner usa AdamW, cross-entropy e gradient
clipping. `encode()` restituisce l'embedding; `forward()` del classificatore
restituisce logits, da convertire con softmax quando servono probabilità.

## Data contract

Ogni riga Parquet rappresenta un evento. I nomi sono esattamente
`{FullReco|L1T}_{collection}_{field}`. Non sono inferiti alias come `JetAK4` per
`JetPuppiAK4` o `MET` per `PUPPIMET`. Il preflight deve confermare che i dati reali
rispettino questo contratto. Gli export scalar-summary storici sono incompatibili.

| Collection | Campi richiesti | Campi opzionali | Feature finali | Limite default |
| --- | --- | --- | ---: | ---: |
| PUPPIPart | PT, Eta, Phi | PID, Charge | 6 | 128 |
| JetPuppiAK4 | PT, Eta, Phi | Mass | 5 | 16 |
| Electron | PT, Eta, Phi | Charge | 5 | 8 |
| MuonTight | PT, Eta, Phi | Charge | 5 | 8 |
| PhotonTight | PT, Eta, Phi | — | 4 | 8 |
| PUPPIMET | MET, Phi | — | 3 | 1 |

Ogni campo è una lista numerica. Sono accettate liste semplici e wrapper esterni
singleton, per esempio `[[10, 20]]`; più vettori annidati ambigui vengono rifiutati.
Campi della stessa collection devono avere lunghezze coerenti. Una lista nulla
o vuota rappresenta zero oggetti; un campo opzionale assente viene riempito con
zero e registrato nei metadati. Una colonna obbligatoria assente è un errore.

Il reader streaming converte le unità dichiarate in GeV e radianti prima delle
trasformazioni. I metadati di unità sono confrontati con il config; se mancano,
il report indica che le unità sono assunte. Le trasformazioni sono:

- PT, MET, Mass: clipping in `[0, value_clip]`, poi `log1p`.
- Eta: clipping in `[-eta_clip, eta_clip]`, poi divisione per `eta_clip`.
- Phi: due feature `sin(phi)`, `cos(phi)`.
- PID: `sign(PID) * log1p(min(abs(PID), value_clip))`.
- Charge: clipping in `[-3, 3]`.

NaN, inf e valori numerici nulli diventano zero; i contatori ne registrano il
numero. Default: `eta_clip=5`, `value_clip=1000000`. Il PID numerico è una scelta
di fattibilità; non è una rappresentazione categorica apprendibile delle specie.

Gli oggetti sono ordinati per PT decrescente prima del troncamento; MET mantiene
l'ordine originale. Il limite zero disabilita una collection. Nel runner streaming
ogni collection abilitata deve avere un limite finito.

## Masks, padding e batch

Le features sono float32; le masks bool con `True` per un oggetto reale. Ogni
collection viene padded fino al massimo nel batch. Tokenizer e Perceiver
escludono il padding anche quando contiene NaN/inf. La somma dei limiti default
è 169 token per evento; le masks indicano quanti sono realmente presenti.

I group IDs sono stabili nell'ordine di configurazione. Il tokenizer accetta un
sottoinsieme delle collections configurate, ma rifiuta gruppi sconosciuti e feature
di dimensione errata. Il reader richiede le colonne di tutte le collections
abilitate; una collection presente ma vuota è valida.

Un evento completamente vuoto usa un sentinel zero interno visibile solo a
quell'evento, evitando softmax completamente mascherate e gradienti NaN.
Il sentinel non è contato nei token del tokenizer. La predizione di un evento
vuoto dipende dai latenti appresi e non contiene informazione osservata.

## Training, test e report

Il solo comando di training è `scripts/run_classification.py`. Un piccolo smoke
è un run con pochi eventi/epoche dello stesso codice usato per il dataset intero.
Il reader streaming mantiene batch e buffer limitati in memoria; i processi,
campioni e split vengono congelati in `dataset_manifest.json`.

`train` include preflight, training, selezione tramite validation, valutazione
test e report. `evaluate` ricarica il checkpoint e usa lo stesso manifest;
`report` legge solo `metrics.json`. Conservare tutti gli artefatti insieme.
I checkpoint del precedente trainer smoke non sono compatibili con il runner
streaming: serve un nuovo run, non basta rinominare il vecchio checkpoint.

I test `test_foundation_vertical_slice.py` coprono entrambi i view, trasformazioni,
troncamento, padding, invarianza al padding/ordine, forward/backward, gradienti
finiti, training CLI, valutazione da checkpoint e report. I test streaming coprono
campionamento, disgiunzione degli split, unità, fingerprint e divisione fra worker.
Il test CLI FullReco usa due worker con `spawn` e verifica anche la chiusura
del processo; L1T copre il caricamento senza multiprocessing.
I test HF simulano lo stream remoto: nessun dataset esterno o accesso EOS.

Per comandi e configurazione vedere il [README](../README.md) e la
[guida ai run](classification_run.md).
