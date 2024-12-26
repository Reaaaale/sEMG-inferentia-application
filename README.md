# Real-Time Data Processing and Inference Demo

## Descrizione del progetto

Questo progetto è una demo sviluppata presso l'Università di Cagliari nell'ambito della borsa di ricerca del progetto europeo "EdgeAI: Edge AI Technologies for Optimised Performance Embedded Processing".

Si tratta di una demo per un applicazione che acquisisce dati sEMG in tempo reale tramite un'interfaccia Arduino con 3 sensori EMG, li filtra e processa, ed esegue inferenza utilizzando un modello di machine learning. 

Il progetto utilizza il framework **Lava-DL** per reti neurali spiking, configurato per lavorare con segnali elaborati in tempo reale. Sono disponibili due modalità di esecuzione:

- **`main_fifo.py`**: Utilizzare questa modalità se non si dispone di una GPU veloce sul PC.
- **`main_threading.py`**: Utilizzare questa modalità se si dispone di una GPU veloce, per migliorare le prestazioni.

---

## Struttura della repository

La repository è organizzata come segue:

- **`src/`**: Codice sorgente con le funzioni principali per il filtraggio, il preprocessing e l'inferenza.
- **`scripts/`**: Script eseguibili per avviare la pipeline.
- **`models/`**: Contiene il modello pre-addestrato utilizzato per l'inferenza.
- **`data/`**: Contiene file di esempio (facoltativo, da includere solo se i dati non sono sensibili).
- **`external/`**: Include file di terze parti, come il framework Lava-DL.

---

## Requisiti

Prima di iniziare, assicurati di avere i seguenti prerequisiti:

- Python 3.10
- `pip` installato

---

## Installazione

### 1. Clona questa repository

```bash
git clone https://github.com/tuo-utente/tuo-repo.git
cd tuo-repo
```

### 2. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 3. Installa Lava-DL

Lava-DL può essere installato seguendo le istruzioni ufficiali disponibili sul [sito ufficiale di Lava](https://lava-nc.org/). Ecco un esempio di installazione tramite `pip`:

```bash
pip install lava-dl
```

Per ulteriori dettagli su configurazioni avanzate, consulta la [documentazione ufficiale](https://lava-nc.org/docs/installation).

---

## Esecuzione

Per avviare il progetto, esegui uno degli script nella cartella `scripts/`. Ad esempio:

```bash
python scripts/main_fifo.py
```

---

## Modello pre-addestrato

Il modello pre-addestrato utilizzato per l'inferenza si trova nella cartella `models/`. Se hai bisogno di aggiornare il modello, puoi sostituirlo con un file `.pt` compatibile.

Per informazioni sull'acquisizione dei dati e sull'addestramento del modello, consulta la repository dedicata disponibile qui: [Repository di acquisizione e addestramento](https://github.com/tuo-utente/repo-acquisizione-addestramento).

Il modello pre-addestrato utilizzato per l'inferenza si trova nella cartella `models/`. 

---


## Disclaimer

Questo repository è una demo attualmente in fase di aggiornamento e potrebbe non essere completo o privo di bug.
È destinato solo a scopi dimostrativi e non deve essere utilizzato per scopi commerciali senza autorizzazione.

