# Calcolo Incentivo all'Esodo - Documentazione

## Panoramica

Questo modulo implementa un calcolatore completo per gli incentivi all'esodo secondo la normativa italiana vigente. Il sistema considera tutte le variabili della legge italiana per il calcolo degli incentivi destinati ai lavoratori in uscita dal mercato del lavoro.

## Caratteristiche Principali

### 1. NASPI (Nuova Assicurazione Sociale per l'Impiego)

Il sistema calcola accuratamente la NASPI considerando:

- **Importo Base**: Calcolato come 75% della retribuzione media mensile fino a €1.250, poi 25% della parte eccedente
- **Tetto Massimo**: €1.550,42 mensili (aggiornato 2026)
- **Durata**: Metà dei mesi contributivi degli ultimi 4 anni, massimo 24 mesi
- **Riduzione Progressive**: 3% mensile dal 4° mese di percezione
- **Contributi Figurativi**: Calcolati automaticamente su 1.4 volte l'importo NASPI

### 2. Costo della Vita Regionale

Il calcolo dell'incentivo è **variabile per regione** considerando il costo della vita specifico:

| Regione | Coefficiente | Note |
|---------|--------------|------|
| Lombardia | 1.15 | Costi più elevati (Milano) |
| Lazio | 1.12 | Roma capitale |
| Trentino-Alto Adige | 1.10 | Alto tenore di vita |
| Emilia-Romagna | 1.08 | Economia sviluppata |
| Veneto | 1.05 | Nordest produttivo |
| ... | ... | ... |
| Molise | 0.86 | Costi più bassi |

### 3. Lavoratori Precoci

Il sistema verifica i requisiti per lavoratori precoci:

- Almeno **41 anni di contributi**
- Almeno **12 mesi di lavoro prima dei 19 anni**
- Bonus del **10%** sull'incentivo se requisiti soddisfatti

### 4. Lavori Usuranti

Identificazione automatica dei lavori usuranti:

- Lavoro notturno
- Catena di montaggio
- Conducente mezzi pesanti
- Cave e tunnel
- Alte temperature
- Lavoro in altezza
- Palombaro

Bonus del **10%** sull'incentivo per lavoratori usuranti con almeno 30 anni di contributi.

### 5. Tipi di Contribuzione

Supporto completo per diverse tipologie:

- **Dipendente Privato**: Aliquota standard INPS (9.19% lavoratore + 30.10% datore)
- **Artigiani**: Aliquota 24%
- **Autonomi**: Aliquota 25.95%
- **Agricoli**: Aliquota 22.14%

### 6. Previdenza Complementare e R.I.T.A.

Il sistema verifica i requisiti per la R.I.T.A. (Rendita Integrativa Temporanea Anticipata):

**Requisiti R.I.T.A.:**
- Età minima: 57 anni (con cessazione lavoro)
- Contributi minimi: 20 anni
- Montante in previdenza complementare
- Alternativa: Disoccupazione da almeno 24 mesi

**Calcoli R.I.T.A.:**
- Rendita mensile stimata
- Integrazione al reddito durante il periodo di attesa pensione
- Massimo 5 anni di erogazione

### 7. APE Sociale (Anticipo Pensionistico)

Parametri considerati:
- Età minima: 63 anni
- Contributi minimi: 30 anni (36 per alcune categorie)
- Importo massimo: €1.500 mensili

## Formula Calcolo Incentivo

L'incentivo all'esodo è calcolato con la seguente formula:

```
Incentivo = (Delta_Retribuzione - Valore_Tempo_Libero) × Coefficiente_Regionale × Bonus_Speciale

Dove:
- Delta_Retribuzione = Retribuzione_Lavorando - NASPI_Totale
- Valore_Tempo_Libero = (Retribuzione_Netta × 0.30) × Coefficiente_Regionale × Durata_Mesi
- Bonus_Speciale = 1.10 se lavoratore precoce o usurante, altrimenti 1.00
```

## Esempi di Calcolo

### Esempio 1: Lavoratore Standard

**Input:**
- Retribuzione: €2.500/mese
- Regione: Lombardia (coeff. 1.15)
- Contributi ultimi 4 anni: 48 mesi
- Età: 55 anni
- Contributi totali: 30 anni

**Output:**
- Durata NASPI: 24 mesi
- NASPI totale: €22.847,66
- Delta retribuzione: €37.152,34
- Valore tempo libero: €14.490,00
- **Incentivo finale: €26.061,69**

### Esempio 2: Lavoratore Precoce con Lavoro Usurante

**Input:**
- Retribuzione: €3.200/mese
- Regione: Lazio (coeff. 1.12)
- Contributi ultimi 4 anni: 48 mesi
- Età: 58 anni
- Contributi totali: 35 anni
- Mesi lavoro prima 19 anni: 12
- Tipo lavoro: Notturno
- Previdenza complementare: €75.000

**Output:**
- Durata NASPI: 24 mesi
- NASPI totale: €23.712,00
- Delta retribuzione: €53.088,00
- Valore tempo libero: €18.662,40
- Bonus speciale: 10%
- R.I.T.A. eligible: Sì
- **Incentivo finale: €42.354,82**

## File Dati di Esempio

### lavoratori_esempio.csv

Contiene dati di esempio per 15 lavoratori con vari profili:
- Diverse regioni di residenza
- Varie tipologie di contribuzione
- Lavoratori precoci e con lavori usuranti
- Con e senza previdenza complementare

### costo_vita_regionale.csv

Dati sul costo della vita per tutte le 20 regioni italiane:
- Coefficiente costo vita
- PIL pro capite
- Costo medio abitazione
- Note specifiche per regione

## Utilizzo dell'Applicazione Streamlit

### Avvio Applicazione

```bash
streamlit run incentivo_esodo.py
```

### Interfaccia Utente

L'applicazione presenta:

**Sidebar - Input Dati:**
- Retribuzione mensile lorda
- Regione di residenza
- Mesi contributi (ultimi 4 anni)
- Anni contributi totali
- Età lavoratore
- Parametri avanzati (tipo contribuzione, tipo lavoro, ecc.)

**Area Principale - Risultati:**

1. **Metriche Principali:**
   - Incentivo esodo finale
   - Delta retribuzione
   - Durata NASPI
   - NASPI totale

2. **Tab Dettagliati:**
   - **Riepilogo**: Sintesi calcolo completo
   - **Dettaglio NASPI**: Tabella e grafico mensile con riduzione
   - **Valore Tempo Libero**: Calcolo dettagliato
   - **Requisiti Speciali**: Verifica lavoratore precoce e usurante
   - **Previdenza Complementare**: R.I.T.A. e montante

## Normativa di Riferimento

Il calcolatore implementa le seguenti normative:

1. **D.Lgs. 22/2015** - Disposizioni NASPI
2. **Legge 232/2016** - APE sociale e precoci
3. **D.Lgs. 252/2005** - Previdenza complementare e R.I.T.A.
4. **D.Lgs. 67/2011** - Lavori usuranti
5. **Prassi INPS** - Circolari e messaggi applicativi

## Limitazioni e Note

- I calcoli sono semplificati per scopi dimostrativi
- I parametri sono aggiornati al 2026 ma possono variare
- Il valore del tempo libero è una stima basata su modelli economici
- Per calcoli ufficiali consultare sempre professionisti abilitati e l'INPS

## Estensioni Future

Possibili miglioramenti:
- Integrazione con API INPS per dati real-time
- Calcolo preciso montante pensionistico
- Simulazione scenari multipli
- Export PDF dei risultati
- Confronto tra diverse opzioni di esodo
- Calcolo età pensionabile con Quota 100/102/103

## Supporto e Contatti

Per domande o segnalazioni:
- Repository: https://github.com/mat635418/Inventorycorridor_ABAR
- Sviluppatore: mat635418
- Data rilascio: Febbraio 2026

---

**DISCLAIMER**: Questo strumento è fornito a scopo informativo e dimostrativo. Per decisioni riguardanti la propria posizione lavorativa e pensionistica, consultare sempre professionisti qualificati e gli uffici INPS competenti.
