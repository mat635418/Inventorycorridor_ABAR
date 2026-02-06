"""
Calcolo Incentivo all'Esodo - Italian Labor Law Exit Incentive Calculator
Considera tutte le variabili della legge italiana per il calcolo degli incentivi all'esodo.

Developed by mat635418 — FEB 2026
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Configurazione pagina
st.set_page_config(
    page_title="Incentivo Esodo - Calcolo",
    layout="wide",
    page_icon="💼"
)

# ============================================================================
# DATI REGIONALI - COSTO DELLA VITA PER REGIONE ITALIANA
# ============================================================================
COSTO_VITA_REGIONALE = {
    "Lombardia": 1.15,
    "Lazio": 1.12,
    "Trentino-Alto Adige": 1.10,
    "Emilia-Romagna": 1.08,
    "Veneto": 1.05,
    "Piemonte": 1.04,
    "Liguria": 1.03,
    "Friuli-Venezia Giulia": 1.02,
    "Toscana": 1.02,
    "Marche": 0.98,
    "Valle d'Aosta": 1.05,
    "Umbria": 0.96,
    "Abruzzo": 0.94,
    "Sardegna": 0.95,
    "Campania": 0.92,
    "Puglia": 0.90,
    "Sicilia": 0.89,
    "Basilicata": 0.88,
    "Calabria": 0.87,
    "Molise": 0.86,
}

# ============================================================================
# PARAMETRI NASPI (Nuova Assicurazione Sociale per l'Impiego)
# ============================================================================
NASPI_TETTO_MASSIMO_MENSILE = 1550.42  # Euro - aggiornato 2026
NASPI_PERCENTUALE_BASE = 0.75  # 75% della retribuzione media
NASPI_PERCENTUALE_ECCEDENZA = 0.25  # 25% per la parte eccedente
NASPI_SOGLIA_CALCOLO = 1250.0  # Euro - soglia per calcolo differenziato
NASPI_RIDUZIONE_MENSILE_DOPO_MESI = 3  # Mesi prima della riduzione
NASPI_RIDUZIONE_PERCENTUALE = 0.03  # 3% mensile di riduzione

# ============================================================================
# PARAMETRI CONTRIBUTIVI
# ============================================================================
ALIQUOTA_CONTRIBUTIVA_INPS_DIPENDENTE = 0.0919  # 9.19% a carico lavoratore
ALIQUOTA_CONTRIBUTIVA_INPS_DATORE = 0.3010  # 30.10% a carico datore
ALIQUOTA_CONTRIBUTIVA_ARTIGIANI = 0.24  # 24% per artigiani
ALIQUOTA_CONTRIBUTIVA_AUTONOMI = 0.2595  # 25.95% per autonomi
ALIQUOTA_CONTRIBUTIVA_AGRICOLI = 0.2214  # 22.14% per agricoli

# ============================================================================
# PARAMETRI PREVIDENZA COMPLEMENTARE
# ============================================================================
TFR_ANNUO_PERCENTUALE = 0.0691  # 6.91% della retribuzione annua
ALIQUOTA_PREVIDENZA_COMPLEMENTARE = 0.015  # 1.5% a carico datore base

# ============================================================================
# RITA (Rendita Integrativa Temporanea Anticipata)
# ============================================================================
RITA_ETA_MINIMA_PENSIONE = 57  # Anni - età minima per RITA con cessazione lavoro
RITA_ANNI_ANTICIPO_MASSIMO = 5  # Anni - anticipo massimo rispetto a pensione vecchiaia
RITA_CONTRIBUTI_MINIMI_ANNI = 20  # Anni minimi di contributi per RITA
RITA_DISOCCUPAZIONE_MESI_MINIMI = 24  # Mesi di disoccupazione da almeno 24 mesi

# ============================================================================
# PARAMETRI LAVORATORI PRECOCI E USURANTI
# ============================================================================
LAVORI_USURANTI_ANTICIPO_ANNI = 5  # Anni di anticipo per lavori usuranti
LAVORATORI_PRECOCI_ETA_MINIMA = 41  # Anni di contributi minimi per precoci
LAVORATORI_PRECOCI_LAVORO_PRIMA_19_ANNI = 12  # Mesi lavoro prima 19 anni

# ============================================================================
# API (Anticipo Pensionistico)
# ============================================================================
APE_SOCIALE_REQUISITI_ANNI = 63  # Età minima per APE sociale
APE_SOCIALE_CONTRIBUTI_ANNI = 30  # Anni minimi di contributi (o 36 per alcuni)
APE_SOCIALE_IMPORTO_MASSIMO = 1500.0  # Euro mensili massimi

# ============================================================================
# FUNZIONI DI CALCOLO
# ============================================================================

def calcola_naspi_mensile(retribuzione_media_mensile):
    """
    Calcola l'importo mensile della NASPI.
    
    Args:
        retribuzione_media_mensile: Retribuzione media mensile degli ultimi 4 anni
        
    Returns:
        Importo NASPI mensile
    """
    if retribuzione_media_mensile <= NASPI_SOGLIA_CALCOLO:
        naspi = retribuzione_media_mensile * NASPI_PERCENTUALE_BASE
    else:
        naspi = (NASPI_SOGLIA_CALCOLO * NASPI_PERCENTUALE_BASE + 
                (retribuzione_media_mensile - NASPI_SOGLIA_CALCOLO) * NASPI_PERCENTUALE_ECCEDENZA)
    
    return min(naspi, NASPI_TETTO_MASSIMO_MENSILE)


def calcola_durata_naspi_mesi(mesi_contributi_ultimi_4_anni):
    """
    Calcola la durata della NASPI in mesi.
    
    Args:
        mesi_contributi_ultimi_4_anni: Mesi di contributi versati negli ultimi 4 anni
        
    Returns:
        Durata NASPI in mesi (massimo 24 mesi)
    """
    durata = min(int(mesi_contributi_ultimi_4_anni / 2), 24)
    return durata


def applica_riduzione_naspi(importo_base, mese_corrente):
    """
    Applica la riduzione mensile del 3% alla NASPI dopo i primi mesi.
    
    Args:
        importo_base: Importo NASPI base
        mese_corrente: Mese corrente di percezione NASPI (1-indexed)
        
    Returns:
        Importo NASPI con riduzione applicata
    """
    if mese_corrente <= NASPI_RIDUZIONE_MENSILE_DOPO_MESI:
        return importo_base
    
    mesi_riduzione = mese_corrente - NASPI_RIDUZIONE_MENSILE_DOPO_MESI
    fattore_riduzione = (1 - NASPI_RIDUZIONE_PERCENTUALE) ** mesi_riduzione
    return importo_base * fattore_riduzione


def calcola_contributi_figurativi_naspi(importo_naspi_mensile):
    """
    Calcola i contributi figurativi accreditati durante la percezione della NASPI.
    
    Args:
        importo_naspi_mensile: Importo mensile NASPI
        
    Returns:
        Contributi figurativi mensili
    """
    # I contributi figurativi sono calcolati su 1.4 volte l'importo NASPI
    base_calcolo = importo_naspi_mensile * 1.4
    contributi_figurativi = base_calcolo * (ALIQUOTA_CONTRIBUTIVA_INPS_DIPENDENTE + 
                                           ALIQUOTA_CONTRIBUTIVA_INPS_DATORE)
    return contributi_figurativi


def calcola_valore_tempo_libero(regione, retribuzione_mensile_lorda):
    """
    Calcola il valore del tempo libero considerando il costo della vita regionale.
    
    Args:
        regione: Regione italiana di residenza
        retribuzione_mensile_lorda: Retribuzione mensile lorda
        
    Returns:
        Valore tempo libero mensile
    """
    coefficiente_regionale = COSTO_VITA_REGIONALE.get(regione, 1.0)
    
    # Il valore del tempo libero è stimato come percentuale della retribuzione
    # adjusted per il costo della vita regionale
    # Assume che il tempo libero valga circa il 30% della retribuzione netta
    # in regioni con costo vita medio, con adjustment per regione
    retribuzione_netta_stimata = retribuzione_mensile_lorda * 0.70  # Stima netto
    valore_tempo_libero_base = retribuzione_netta_stimata * 0.30
    
    # Adjustment per costo vita: maggiore costo vita = maggior valore tempo libero
    valore_tempo_libero = valore_tempo_libero_base * coefficiente_regionale
    
    return valore_tempo_libero


def verifica_requisiti_lavoratore_precoce(anni_contributi, mesi_lavoro_prima_19):
    """
    Verifica se il lavoratore soddisfa i requisiti per lavoratore precoce.
    
    Args:
        anni_contributi: Anni totali di contributi
        mesi_lavoro_prima_19: Mesi di lavoro effettuati prima dei 19 anni
        
    Returns:
        Boolean - True se è lavoratore precoce
    """
    return (anni_contributi >= LAVORATORI_PRECOCI_ETA_MINIMA and 
            mesi_lavoro_prima_19 >= LAVORATORI_PRECOCI_LAVORO_PRIMA_19_ANNI)


def verifica_requisiti_lavoro_usurante(tipo_lavoro, anni_contributi):
    """
    Verifica se il lavoro è classificato come usurante.
    
    Args:
        tipo_lavoro: Tipo di lavoro svolto
        anni_contributi: Anni di contributi
        
    Returns:
        Boolean - True se è lavoro usurante e requisiti soddisfatti
    """
    lavori_usuranti = [
        "lavoro_notturno",
        "catena_montaggio",
        "conducente_mezzi_pesanti",
        "cave_tunnel",
        "alte_temperature",
        "palombaro",
        "lavori_altezza"
    ]
    
    return tipo_lavoro in lavori_usuranti and anni_contributi >= 30


def calcola_previdenza_complementare_accumulata(anni_versamento, contributo_annuo):
    """
    Calcola il montante accumulato nella previdenza complementare.
    
    Args:
        anni_versamento: Anni di versamento
        contributo_annuo: Contributo annuo medio
        
    Returns:
        Montante accumulato (semplificato, senza rendimenti)
    """
    # Calcolo semplificato - andrebbe usato un tasso di rendimento
    montante = anni_versamento * contributo_annuo
    # Aggiunge un rendimento stimato del 2% annuo composto
    montante_con_rendimenti = contributo_annuo * (((1.02 ** anni_versamento) - 1) / 0.02)
    return montante_con_rendimenti


def verifica_requisiti_rita(eta_lavoratore, anni_contributi_totali, 
                            mesi_disoccupazione, montante_previdenza_complementare):
    """
    Verifica i requisiti per accedere alla RITA.
    
    Args:
        eta_lavoratore: Età del lavoratore
        anni_contributi_totali: Anni totali di contributi
        mesi_disoccupazione: Mesi di disoccupazione (se applicabile)
        montante_previdenza_complementare: Montante previdenza complementare
        
    Returns:
        Tuple (eligible, reason)
    """
    if montante_previdenza_complementare <= 0:
        return False, "Nessun montante in previdenza complementare"
    
    if anni_contributi_totali < RITA_CONTRIBUTI_MINIMI_ANNI:
        return False, f"Contributi insufficienti (minimo {RITA_CONTRIBUTI_MINIMI_ANNI} anni)"
    
    # Caso 1: Cessazione attività lavorativa a 5 anni dalla pensione
    eta_pensione_vecchiaia = 67  # Età pensione vecchiaia attuale
    anni_mancanti_pensione = eta_pensione_vecchiaia - eta_lavoratore
    
    if 0 < anni_mancanti_pensione <= RITA_ANNI_ANTICIPO_MASSIMO:
        return True, "Requisiti soddisfatti - cessazione attività"
    
    # Caso 2: Disoccupazione da almeno 24 mesi
    if mesi_disoccupazione >= RITA_DISOCCUPAZIONE_MESI_MINIMI and eta_lavoratore >= RITA_ETA_MINIMA_PENSIONE:
        return True, "Requisiti soddisfatti - disoccupazione prolungata"
    
    return False, "Requisiti non soddisfatti"


def calcola_incentivo_esodo(
    retribuzione_mensile_lorda,
    mesi_contributi_ultimi_4_anni,
    regione_residenza,
    anni_contributi_totali=30,
    tipo_contribuzione="dipendente_privato",
    tipo_lavoro="standard",
    mesi_lavoro_prima_19=0,
    eta_lavoratore=55,
    ha_previdenza_complementare=False,
    montante_previdenza_complementare=0,
    mesi_disoccupazione=0
):
    """
    Calcola l'incentivo all'esodo completo secondo la normativa italiana.
    
    Args:
        retribuzione_mensile_lorda: Retribuzione mensile lorda attuale
        mesi_contributi_ultimi_4_anni: Mesi di contributi negli ultimi 4 anni
        regione_residenza: Regione di residenza
        anni_contributi_totali: Anni totali di contributi
        tipo_contribuzione: Tipo di contribuzione (dipendente_privato, artigiano, autonomo, agricolo)
        tipo_lavoro: Tipo di lavoro (standard, lavoro_notturno, etc.)
        mesi_lavoro_prima_19: Mesi lavorati prima dei 19 anni
        eta_lavoratore: Età del lavoratore
        ha_previdenza_complementare: Se ha previdenza complementare
        montante_previdenza_complementare: Montante previdenza complementare
        mesi_disoccupazione: Mesi di disoccupazione (se applicabile)
        
    Returns:
        Dictionary con tutti i calcoli dettagliati
    """
    risultato = {
        "dati_input": {
            "retribuzione_mensile_lorda": retribuzione_mensile_lorda,
            "regione_residenza": regione_residenza,
            "coefficiente_costo_vita": COSTO_VITA_REGIONALE.get(regione_residenza, 1.0),
            "mesi_contributi_ultimi_4_anni": mesi_contributi_ultimi_4_anni,
            "anni_contributi_totali": anni_contributi_totali,
            "tipo_contribuzione": tipo_contribuzione,
            "tipo_lavoro": tipo_lavoro,
            "eta_lavoratore": eta_lavoratore,
        },
        "naspi": {},
        "contributi_figurativi": {},
        "valore_tempo_libero": {},
        "incentivo_esodo": {},
        "requisiti_speciali": {},
        "previdenza_complementare": {},
    }
    
    # 1. Calcolo NASPI
    naspi_base = calcola_naspi_mensile(retribuzione_mensile_lorda)
    durata_naspi = calcola_durata_naspi_mesi(mesi_contributi_ultimi_4_anni)
    
    risultato["naspi"]["importo_mensile_base"] = naspi_base
    risultato["naspi"]["durata_mesi"] = durata_naspi
    
    # Calcolo NASPI per ogni mese con riduzione
    importi_naspi_mensili = []
    for mese in range(1, durata_naspi + 1):
        importo_mese = applica_riduzione_naspi(naspi_base, mese)
        importi_naspi_mensili.append(importo_mese)
    
    risultato["naspi"]["importi_mensili"] = importi_naspi_mensili
    risultato["naspi"]["totale_naspi_periodo"] = sum(importi_naspi_mensili)
    
    # 2. Calcolo contributi figurativi
    contributi_figurativi_mensili = []
    for importo_naspi in importi_naspi_mensili:
        contrib_fig = calcola_contributi_figurativi_naspi(importo_naspi)
        contributi_figurativi_mensili.append(contrib_fig)
    
    risultato["contributi_figurativi"]["mensili"] = contributi_figurativi_mensili
    risultato["contributi_figurativi"]["totale"] = sum(contributi_figurativi_mensili)
    
    # 3. Calcolo retribuzione che avrebbe percepito lavorando
    retribuzione_totale_lavorando = retribuzione_mensile_lorda * durata_naspi
    risultato["incentivo_esodo"]["retribuzione_se_lavorasse"] = retribuzione_totale_lavorando
    
    # 4. Calcolo delta (differenza tra retribuzione e NASPI)
    delta_retribuzione = retribuzione_totale_lavorando - risultato["naspi"]["totale_naspi_periodo"]
    risultato["incentivo_esodo"]["delta_retribuzione"] = delta_retribuzione
    
    # 5. Calcolo valore tempo libero
    valore_tempo_libero_mensile = calcola_valore_tempo_libero(
        regione_residenza, 
        retribuzione_mensile_lorda
    )
    valore_tempo_libero_totale = valore_tempo_libero_mensile * durata_naspi
    
    risultato["valore_tempo_libero"]["mensile"] = valore_tempo_libero_mensile
    risultato["valore_tempo_libero"]["totale_periodo"] = valore_tempo_libero_totale
    
    # 6. Calcolo incentivo esodo finale
    # L'incentivo deve coprire la differenza tra retribuzione e NASPI
    # Considerando anche il valore del tempo libero che viene "guadagnato"
    incentivo_esodo_base = delta_retribuzione - valore_tempo_libero_totale
    
    # Applica coefficiente regionale all'incentivo
    coefficiente_regionale = COSTO_VITA_REGIONALE.get(regione_residenza, 1.0)
    incentivo_esodo_adjusted = incentivo_esodo_base * coefficiente_regionale
    
    risultato["incentivo_esodo"]["incentivo_base"] = incentivo_esodo_base
    risultato["incentivo_esodo"]["incentivo_adjusted_regionale"] = incentivo_esodo_adjusted
    risultato["incentivo_esodo"]["coefficiente_regionale"] = coefficiente_regionale
    
    # 7. Verifica requisiti speciali
    is_precoce = verifica_requisiti_lavoratore_precoce(anni_contributi_totali, mesi_lavoro_prima_19)
    is_usurante = verifica_requisiti_lavoro_usurante(tipo_lavoro, anni_contributi_totali)
    
    risultato["requisiti_speciali"]["lavoratore_precoce"] = is_precoce
    risultato["requisiti_speciali"]["lavoro_usurante"] = is_usurante
    
    # 8. Previdenza complementare e RITA
    if ha_previdenza_complementare:
        rita_eligible, rita_reason = verifica_requisiti_rita(
            eta_lavoratore,
            anni_contributi_totali,
            mesi_disoccupazione,
            montante_previdenza_complementare
        )
        
        risultato["previdenza_complementare"]["montante"] = montante_previdenza_complementare
        risultato["previdenza_complementare"]["rita_eligible"] = rita_eligible
        risultato["previdenza_complementare"]["rita_reason"] = rita_reason
        
        if rita_eligible:
            # RITA può fornire reddito aggiuntivo durante il periodo di attesa pensione
            anni_erogazione_rita = min(5, 67 - eta_lavoratore)  # Massimo 5 anni
            if anni_erogazione_rita > 0:
                rita_mensile_stimata = (montante_previdenza_complementare / 
                                       (anni_erogazione_rita * 12))
                risultato["previdenza_complementare"]["rita_mensile_stimata"] = rita_mensile_stimata
                risultato["previdenza_complementare"]["rita_totale_periodo"] = (
                    rita_mensile_stimata * min(durata_naspi, anni_erogazione_rita * 12)
                )
    
    # 9. Calcolo finale considerando tutti i fattori
    incentivo_finale = incentivo_esodo_adjusted
    
    # Aggiungi bonus per lavoratori precoci o usuranti
    if is_precoce or is_usurante:
        bonus_percentuale = 0.10  # 10% bonus
        incentivo_finale = incentivo_finale * (1 + bonus_percentuale)
        risultato["incentivo_esodo"]["bonus_speciale"] = incentivo_esodo_adjusted * bonus_percentuale
    
    risultato["incentivo_esodo"]["incentivo_finale"] = incentivo_finale
    
    return risultato


# ============================================================================
# INTERFACCIA STREAMLIT
# ============================================================================

def main():
    """Interfaccia principale Streamlit per il calcolatore incentivo esodo."""
    
    st.title("💼 Calcolo Incentivo all'Esodo")
    st.markdown("""
    ### Calcolo secondo la normativa italiana vigente
    Questo strumento calcola l'incentivo all'esodo considerando:
    - NASPI con copertura figurativa contributi
    - Costo della vita per regione
    - Lavoratori precoci e lavori usuranti
    - Previdenza complementare e R.I.T.A.
    - Tutte le variabili della legge italiana
    """)
    
    st.divider()
    
    # Sidebar per input
    with st.sidebar:
        st.header("📋 Dati Lavoratore")
        
        retribuzione = st.number_input(
            "Retribuzione Mensile Lorda (€)",
            min_value=800.0,
            max_value=10000.0,
            value=2500.0,
            step=100.0
        )
        
        regione = st.selectbox(
            "Regione di Residenza",
            options=sorted(COSTO_VITA_REGIONALE.keys())
        )
        
        mesi_contributi = st.slider(
            "Mesi Contributi (ultimi 4 anni)",
            min_value=13,
            max_value=48,
            value=48
        )
        
        anni_contributi = st.slider(
            "Anni Contributi Totali",
            min_value=5,
            max_value=45,
            value=30
        )
        
        eta = st.slider(
            "Età Lavoratore",
            min_value=40,
            max_value=67,
            value=55
        )
        
        st.divider()
        st.subheader("🔧 Parametri Avanzati")
        
        tipo_contrib = st.selectbox(
            "Tipo Contribuzione",
            options=[
                "dipendente_privato",
                "artigiano",
                "autonomo",
                "agricolo"
            ]
        )
        
        tipo_lav = st.selectbox(
            "Tipo Lavoro",
            options=[
                "standard",
                "lavoro_notturno",
                "catena_montaggio",
                "conducente_mezzi_pesanti",
                "cave_tunnel",
                "alte_temperature"
            ]
        )
        
        mesi_prima_19 = st.number_input(
            "Mesi Lavoro Prima 19 Anni",
            min_value=0,
            max_value=36,
            value=0
        )
        
        ha_prev_compl = st.checkbox("Ha Previdenza Complementare")
        
        montante = 0
        mesi_disoc = 0
        if ha_prev_compl:
            montante = st.number_input(
                "Montante Previdenza Complementare (€)",
                min_value=0.0,
                max_value=500000.0,
                value=50000.0,
                step=5000.0
            )
            mesi_disoc = st.number_input(
                "Mesi Disoccupazione (se applicabile)",
                min_value=0,
                max_value=60,
                value=0
            )
        
        calcola_btn = st.button("🧮 Calcola Incentivo", type="primary", use_container_width=True)
    
    # Area principale - risultati
    if calcola_btn:
        with st.spinner("Calcolo in corso..."):
            risultato = calcola_incentivo_esodo(
                retribuzione_mensile_lorda=retribuzione,
                mesi_contributi_ultimi_4_anni=mesi_contributi,
                regione_residenza=regione,
                anni_contributi_totali=anni_contributi,
                tipo_contribuzione=tipo_contrib,
                tipo_lavoro=tipo_lav,
                mesi_lavoro_prima_19=mesi_prima_19,
                eta_lavoratore=eta,
                ha_previdenza_complementare=ha_prev_compl,
                montante_previdenza_complementare=montante,
                mesi_disoccupazione=mesi_disoc
            )
        
        # Mostra risultati
        st.success("✅ Calcolo completato!")
        
        # Metriche principali
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Incentivo Esodo Finale",
                f"€ {risultato['incentivo_esodo']['incentivo_finale']:,.2f}",
                delta=f"Coeff. Reg. {risultato['incentivo_esodo']['coefficiente_regionale']:.2f}"
            )
        
        with col2:
            st.metric(
                "Delta Retribuzione",
                f"€ {risultato['incentivo_esodo']['delta_retribuzione']:,.2f}"
            )
        
        with col3:
            st.metric(
                "Durata NASPI",
                f"{risultato['naspi']['durata_mesi']} mesi"
            )
        
        with col4:
            st.metric(
                "NASPI Totale",
                f"€ {risultato['naspi']['totale_naspi_periodo']:,.2f}"
            )
        
        st.divider()
        
        # Tabs per dettagli
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Riepilogo",
            "💰 Dettaglio NASPI",
            "🎯 Valore Tempo Libero",
            "🏆 Requisiti Speciali",
            "📈 Previdenza Complementare"
        ])
        
        with tab1:
            st.subheader("Riepilogo Calcolo Incentivo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dati Input**")
                st.write(f"- Retribuzione mensile: € {risultato['dati_input']['retribuzione_mensile_lorda']:,.2f}")
                st.write(f"- Regione: {risultato['dati_input']['regione_residenza']}")
                st.write(f"- Coefficiente costo vita: {risultato['dati_input']['coefficiente_costo_vita']:.2f}")
                st.write(f"- Età: {risultato['dati_input']['eta_lavoratore']} anni")
                st.write(f"- Contributi totali: {risultato['dati_input']['anni_contributi_totali']} anni")
            
            with col2:
                st.write("**Calcolo Incentivo**")
                st.write(f"- Retribuzione se lavorasse: € {risultato['incentivo_esodo']['retribuzione_se_lavorasse']:,.2f}")
                st.write(f"- NASPI totale percepita: € {risultato['naspi']['totale_naspi_periodo']:,.2f}")
                st.write(f"- Delta retribuzione: € {risultato['incentivo_esodo']['delta_retribuzione']:,.2f}")
                st.write(f"- Valore tempo libero: € {risultato['valore_tempo_libero']['totale_periodo']:,.2f}")
                st.write(f"- **Incentivo finale: € {risultato['incentivo_esodo']['incentivo_finale']:,.2f}**")
        
        with tab2:
            st.subheader("Dettaglio NASPI Mensile")
            
            df_naspi = pd.DataFrame({
                "Mese": range(1, len(risultato['naspi']['importi_mensili']) + 1),
                "NASPI (€)": risultato['naspi']['importi_mensili'],
                "Contributi Figurativi (€)": risultato['contributi_figurativi']['mensili']
            })
            
            st.dataframe(df_naspi, use_container_width=True)
            
            st.write(f"**Totale NASPI:** € {risultato['naspi']['totale_naspi_periodo']:,.2f}")
            st.write(f"**Totale Contributi Figurativi:** € {risultato['contributi_figurativi']['totale']:,.2f}")
            
            # Grafico NASPI
            import plotly.express as px
            fig = px.line(
                df_naspi, 
                x="Mese", 
                y="NASPI (€)",
                title="Andamento NASPI Mensile (con riduzione 3% dal 4° mese)",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Valore del Tempo Libero")
            
            st.write(f"""
            Il valore del tempo libero è calcolato considerando:
            - La retribuzione netta stimata
            - Il coefficiente del costo della vita regionale ({risultato['dati_input']['coefficiente_costo_vita']:.2f})
            - Un valore stimato del 30% della retribuzione netta
            """)
            
            st.metric(
                "Valore Tempo Libero Mensile",
                f"€ {risultato['valore_tempo_libero']['mensile']:,.2f}"
            )
            
            st.metric(
                "Valore Tempo Libero Totale Periodo",
                f"€ {risultato['valore_tempo_libero']['totale_periodo']:,.2f}"
            )
        
        with tab4:
            st.subheader("Requisiti Speciali")
            
            is_precoce = risultato['requisiti_speciali']['lavoratore_precoce']
            is_usurante = risultato['requisiti_speciali']['lavoro_usurante']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_precoce:
                    st.success("✅ Lavoratore Precoce")
                else:
                    st.info("ℹ️ Non Lavoratore Precoce")
            
            with col2:
                if is_usurante:
                    st.success("✅ Lavoro Usurante")
                else:
                    st.info("ℹ️ Lavoro Non Usurante")
            
            if is_precoce or is_usurante:
                st.write("**Bonus Speciale Applicato:** 10%")
                if "bonus_speciale" in risultato['incentivo_esodo']:
                    st.write(f"Importo bonus: € {risultato['incentivo_esodo']['bonus_speciale']:,.2f}")
        
        with tab5:
            st.subheader("Previdenza Complementare e R.I.T.A.")
            
            if ha_prev_compl:
                st.write(f"**Montante Previdenza Complementare:** € {risultato['previdenza_complementare']['montante']:,.2f}")
                
                rita_eligible = risultato['previdenza_complementare']['rita_eligible']
                rita_reason = risultato['previdenza_complementare']['rita_reason']
                
                if rita_eligible:
                    st.success(f"✅ Requisiti R.I.T.A. soddisfatti: {rita_reason}")
                    
                    if "rita_mensile_stimata" in risultato['previdenza_complementare']:
                        st.metric(
                            "R.I.T.A. Mensile Stimata",
                            f"€ {risultato['previdenza_complementare']['rita_mensile_stimata']:,.2f}"
                        )
                        st.metric(
                            "R.I.T.A. Totale Periodo",
                            f"€ {risultato['previdenza_complementare']['rita_totale_periodo']:,.2f}"
                        )
                else:
                    st.warning(f"⚠️ Requisiti R.I.T.A. non soddisfatti: {rita_reason}")
            else:
                st.info("ℹ️ Nessuna previdenza complementare dichiarata")


if __name__ == "__main__":
    main()
