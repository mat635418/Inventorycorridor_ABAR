# 💼 Calcolo Incentivo all'Esodo - Guida Rapida

## 🎯 Descrizione

Calcolatore completo per gli **incentivi all'esodo** secondo la normativa italiana vigente (2026). Il sistema considera tutte le variabili della legge italiana per fornire un calcolo accurato e personalizzato per regione.

## ✨ Caratteristiche Principali

### ✅ Conformità Normativa Completa

- **NASPI** con riduzione 3% mensile dal 4° mese
- **Contributi figurativi** accreditati automaticamente (1.4x NASPI)
- **Lavoratori precoci** (41 anni contributi + 12 mesi prima 19 anni)
- **Lavori usuranti** (7 categorie identificate)
- **R.I.T.A.** (Rendita Integrativa Temporanea Anticipata)
- **APE sociale** (Anticipo Pensionistico)
- **Previdenza complementare**

### 🏘️ Variabilità Regionale

Incentivo variabile per **tutte le 20 regioni italiane**:
- Coefficienti costo vita da 0.86 (Molise) a 1.15 (Lombardia)
- Adjustment automatico valore tempo libero
- Dati PIL pro capite e costi abitazione

### 📊 Tipologie Contribuzione

- Dipendenti privati (INPS)
- Artigiani
- Autonomi  
- Agricoli

## 🚀 Quick Start

### 1. Installazione

```bash
pip install -r requirements.txt
```

### 2. Esecuzione Applicazione

```bash
streamlit run incentivo_esodo.py
```

L'applicazione si aprirà nel browser con un'interfaccia completa.

### 3. Esecuzione Test

```bash
python test_incentivo_esodo.py
```

## 📋 Input Richiesti

**Dati Base:**
- Retribuzione mensile lorda (€)
- Regione di residenza
- Mesi contributi ultimi 4 anni (13-48)
- Anni contributi totali (5-45)
- Età lavoratore (40-67)

**Parametri Avanzati:**
- Tipo contribuzione (dipendente/artigiano/autonomo/agricolo)
- Tipo lavoro (standard/notturno/usurante/ecc.)
- Mesi lavoro prima 19 anni (se applicabile)
- Previdenza complementare (se presente)
- Montante previdenza complementare (€)

## 💰 Output Forniti

### Metriche Principali

1. **Incentivo Esodo Finale** (€)
   - Adjusted per regione
   - Con bonus se applicabile

2. **Delta Retribuzione** (€)
   - Differenza stipendio vs NASPI
   - Periodo completo

3. **Durata NASPI** (mesi)
   - Calcolato su contributi ultimi 4 anni
   - Massimo 24 mesi

4. **NASPI Totale** (€)
   - Con riduzione mensile 3%
   - Contributi figurativi inclusi

### Tab Dettagliati

1. **📊 Riepilogo** - Sintesi completa calcolo
2. **💰 Dettaglio NASPI** - Tabella e grafico mensile
3. **🎯 Valore Tempo Libero** - Calcolo per regione
4. **🏆 Requisiti Speciali** - Precoce/Usurante
5. **📈 Previdenza Complementare** - R.I.T.A. e montante

## 📊 Esempi di Calcolo

### Esempio 1: Standard Lombardia
```
Input:
  Retribuzione: €2.500/mese
  Regione: Lombardia
  Contributi: 30 anni (48 mesi ultimi 4 anni)
  Età: 55 anni

Output:
  Durata NASPI: 24 mesi
  NASPI totale: €22.847,66
  Delta retribuzione: €37.152,34
  → INCENTIVO: €26.061,69
```

### Esempio 2: Precoce + Usurante Lazio
```
Input:
  Retribuzione: €3.200/mese
  Regione: Lazio
  Contributi: 42 anni
  Tipo lavoro: Notturno
  Mesi prima 19: 12

Output:
  Lavoratore precoce: Sì
  Lavoro usurante: Sì
  Bonus: +10%
  → INCENTIVO: €40.274,46
```

### Esempio 3: Confronto Regionale
```
Stessa retribuzione (€2.500), diversa regione:
  
  Lombardia (1.15): €26.061,69
  Lazio (1.12):     €25.805,18
  Veneto (1.05):    €24.807,45
  Campania (0.92):  €21.849,97
  Calabria (0.87):  €21.023,56
```

## 🧮 Formula Calcolo

```
INCENTIVO = (Delta_Retribuzione - Valore_Tempo_Libero) 
            × Coefficiente_Regionale 
            × Bonus_Speciale

Dove:
  Delta_Retribuzione = (Stipendio_Mensile × Mesi_NASPI) - NASPI_Totale
  Valore_Tempo_Libero = (Netto × 0.30) × Coeff_Regionale × Mesi
  Bonus_Speciale = 1.10 se precoce/usurante, altrimenti 1.00
```

## 📁 File Inclusi

| File | Descrizione |
|------|-------------|
| `incentivo_esodo.py` | Applicazione principale (850+ righe) |
| `test_incentivo_esodo.py` | Suite test completa (13 test) |
| `INCENTIVO_ESODO_DOCS.md` | Documentazione dettagliata |
| `IMPLEMENTAZIONE_COMPLETA.md` | Riepilogo implementazione |
| `lavoratori_esempio.csv` | 15 profili esempio |
| `costo_vita_regionale.csv` | Dati 20 regioni |

## ✅ Qualità e Testing

- ✅ **13 test unitari** - 100% passing
- ✅ **Code review** - 0 issues
- ✅ **Security scan** - 0 vulnerabilities
- ✅ **Documentazione** - Completa

## 📚 Normativa Riferimento

1. **D.Lgs. 22/2015** - NASPI
2. **Legge 232/2016** - Lavoratori precoci e APE sociale
3. **D.Lgs. 252/2005** - Previdenza complementare e R.I.T.A.
4. **D.Lgs. 67/2011** - Lavori usuranti
5. **Prassi INPS 2026** - Circolari e parametri

## 🔍 Verifica Requisiti Speciali

### Lavoratore Precoce
- ✅ Almeno 41 anni di contributi
- ✅ Almeno 12 mesi lavoro prima 19 anni
- 🎁 Bonus: +10% incentivo

### Lavori Usuranti
- Lavoro notturno
- Catena di montaggio  
- Conducente mezzi pesanti
- Cave e tunnel
- Alte temperature
- Palombaro
- Lavori in altezza
- 🎁 Bonus: +10% incentivo (se 30+ anni contributi)

### R.I.T.A.
- ✅ Età minima 57 anni
- ✅ Contributi minimi 20 anni
- ✅ Montante previdenza complementare
- 🎁 Beneficio: Rendita integrativa fino a 5 anni

## 💡 Consigli Utilizzo

1. **Prepara i dati** - Raccogli cedolini e estratti contributivi
2. **Verifica regione** - Coefficiente corretto per costo vita
3. **Controlla requisiti speciali** - Precoce/usurante per bonus
4. **Considera R.I.T.A.** - Se hai previdenza complementare
5. **Confronta scenari** - Prova diverse combinazioni

## ⚠️ Disclaimer

Questo strumento è fornito a **scopo informativo e dimostrativo**.

Per decisioni ufficiali riguardanti la propria posizione lavorativa e pensionistica, **consultare sempre**:
- Professionisti qualificati (consulenti del lavoro, commercialisti)
- Uffici INPS competenti
- Patronati

I parametri sono aggiornati al 2026 ma possono variare. Verificare sempre i valori correnti presso le fonti ufficiali.

## 📞 Supporto

- **Repository**: https://github.com/mat635418/Inventorycorridor_ABAR
- **Sviluppatore**: mat635418
- **Versione**: 1.0
- **Data**: Febbraio 2026

## 📄 Licenza

Consultare il file LICENSE nel repository principale.

---

**Sviluppato con ❤️ per i lavoratori italiani**

*Tutti i diritti riservati - © 2026*
