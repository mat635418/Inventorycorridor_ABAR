"""
Test suite per il calcolatore Incentivo Esodo
Verifica tutte le funzionalità principali del sistema
"""

from incentivo_esodo import (
    calcola_naspi_mensile,
    calcola_durata_naspi_mesi,
    applica_riduzione_naspi,
    calcola_contributi_figurativi_naspi,
    calcola_valore_tempo_libero,
    verifica_requisiti_lavoratore_precoce,
    verifica_requisiti_lavoro_usurante,
    verifica_requisiti_rita,
    calcola_incentivo_esodo,
    COSTO_VITA_REGIONALE,
    NASPI_TETTO_MASSIMO_MENSILE,
    NASPI_SOGLIA_CALCOLO,
)


def test_naspi_sotto_soglia():
    """Test calcolo NASPI per retribuzione sotto soglia"""
    retribuzione = 1000.0
    naspi = calcola_naspi_mensile(retribuzione)
    atteso = 1000.0 * 0.75
    assert naspi == atteso, f"NASPI sotto soglia: atteso {atteso}, ottenuto {naspi}"
    print(f"✓ Test NASPI sotto soglia: €{naspi:.2f}")


def test_naspi_sopra_soglia():
    """Test calcolo NASPI per retribuzione sopra soglia"""
    retribuzione = 2500.0
    naspi = calcola_naspi_mensile(retribuzione)
    # 1250 * 0.75 + (2500-1250) * 0.25 = 937.5 + 312.5 = 1250
    atteso = NASPI_SOGLIA_CALCOLO * 0.75 + (retribuzione - NASPI_SOGLIA_CALCOLO) * 0.25
    assert naspi == atteso, f"NASPI sopra soglia: atteso {atteso}, ottenuto {naspi}"
    print(f"✓ Test NASPI sopra soglia: €{naspi:.2f}")


def test_naspi_tetto_massimo():
    """Test che NASPI non superi il tetto massimo"""
    retribuzione = 5000.0
    naspi = calcola_naspi_mensile(retribuzione)
    assert naspi <= NASPI_TETTO_MASSIMO_MENSILE, f"NASPI supera tetto: {naspi} > {NASPI_TETTO_MASSIMO_MENSILE}"
    assert naspi == NASPI_TETTO_MASSIMO_MENSILE, f"NASPI dovrebbe essere al tetto: {naspi}"
    print(f"✓ Test NASPI tetto massimo: €{naspi:.2f}")


def test_durata_naspi():
    """Test calcolo durata NASPI"""
    # 48 mesi -> 24 mesi NASPI
    durata = calcola_durata_naspi_mesi(48)
    assert durata == 24, f"Durata NASPI: atteso 24, ottenuto {durata}"
    
    # 30 mesi -> 15 mesi NASPI
    durata = calcola_durata_naspi_mesi(30)
    assert durata == 15, f"Durata NASPI: atteso 15, ottenuto {durata}"
    
    # 50 mesi -> 24 mesi NASPI (max)
    durata = calcola_durata_naspi_mesi(50)
    assert durata == 24, f"Durata NASPI max: atteso 24, ottenuto {durata}"
    
    print(f"✓ Test durata NASPI: OK")


def test_riduzione_naspi():
    """Test riduzione NASPI dal 4° mese"""
    importo_base = 1000.0
    
    # Primi 3 mesi: nessuna riduzione
    for mese in range(1, 4):
        importo = applica_riduzione_naspi(importo_base, mese)
        assert importo == importo_base, f"Mese {mese}: non dovrebbe esserci riduzione"
    
    # 4° mese: riduzione 3%
    importo_mese_4 = applica_riduzione_naspi(importo_base, 4)
    atteso_mese_4 = importo_base * 0.97
    assert abs(importo_mese_4 - atteso_mese_4) < 0.01, f"Mese 4: atteso {atteso_mese_4}, ottenuto {importo_mese_4}"
    
    # 5° mese: riduzione 6%
    importo_mese_5 = applica_riduzione_naspi(importo_base, 5)
    atteso_mese_5 = importo_base * (0.97 ** 2)
    assert abs(importo_mese_5 - atteso_mese_5) < 0.01, f"Mese 5: atteso {atteso_mese_5}, ottenuto {importo_mese_5}"
    
    print(f"✓ Test riduzione NASPI: OK")


def test_contributi_figurativi():
    """Test calcolo contributi figurativi"""
    naspi_mensile = 1000.0
    contrib = calcola_contributi_figurativi_naspi(naspi_mensile)
    # Base: 1000 * 1.4 = 1400
    # Contributi: 1400 * (0.0919 + 0.3010) = 1400 * 0.3929 = 550.06
    base_calcolo = naspi_mensile * 1.4
    atteso = base_calcolo * (0.0919 + 0.3010)
    assert abs(contrib - atteso) < 0.01, f"Contributi figurativi: atteso {atteso}, ottenuto {contrib}"
    print(f"✓ Test contributi figurativi: €{contrib:.2f}")


def test_costo_vita_regionale():
    """Test coefficienti costo vita regionali"""
    # Lombardia: costo vita più alto
    valore_lombardia = calcola_valore_tempo_libero("Lombardia", 2500.0)
    
    # Calabria: costo vita più basso
    valore_calabria = calcola_valore_tempo_libero("Calabria", 2500.0)
    
    assert valore_lombardia > valore_calabria, "Lombardia dovrebbe avere valore tempo libero maggiore di Calabria"
    
    # Verifica che tutti i coefficienti siano nel range ragionevole
    for regione, coeff in COSTO_VITA_REGIONALE.items():
        assert 0.8 <= coeff <= 1.2, f"Coefficiente {regione} fuori range: {coeff}"
    
    print(f"✓ Test costo vita regionale: Lombardia €{valore_lombardia:.2f}, Calabria €{valore_calabria:.2f}")


def test_lavoratore_precoce():
    """Test verifica lavoratore precoce"""
    # Soddisfa requisiti
    is_precoce = verifica_requisiti_lavoratore_precoce(41, 12)
    assert is_precoce == True, "Dovrebbe essere lavoratore precoce"
    
    # Non soddisfa requisiti (anni insufficienti)
    is_precoce = verifica_requisiti_lavoratore_precoce(40, 12)
    assert is_precoce == False, "Non dovrebbe essere lavoratore precoce (anni insufficienti)"
    
    # Non soddisfa requisiti (mesi prima 19 insufficienti)
    is_precoce = verifica_requisiti_lavoratore_precoce(41, 11)
    assert is_precoce == False, "Non dovrebbe essere lavoratore precoce (mesi insufficienti)"
    
    print(f"✓ Test lavoratore precoce: OK")


def test_lavoro_usurante():
    """Test verifica lavoro usurante"""
    # Lavoro usurante con requisiti
    is_usurante = verifica_requisiti_lavoro_usurante("lavoro_notturno", 30)
    assert is_usurante == True, "Dovrebbe essere lavoro usurante"
    
    # Lavoro usurante senza requisiti anni
    is_usurante = verifica_requisiti_lavoro_usurante("lavoro_notturno", 29)
    assert is_usurante == False, "Non dovrebbe essere lavoro usurante (anni insufficienti)"
    
    # Lavoro non usurante
    is_usurante = verifica_requisiti_lavoro_usurante("standard", 35)
    assert is_usurante == False, "Lavoro standard non è usurante"
    
    print(f"✓ Test lavoro usurante: OK")


def test_requisiti_rita():
    """Test verifica requisiti R.I.T.A."""
    # Caso 1: Cessazione attività a 5 anni dalla pensione
    eligible, reason = verifica_requisiti_rita(62, 25, 0, 50000)
    assert eligible == True, f"Dovrebbe essere eligible per R.I.T.A.: {reason}"
    
    # Caso 2: Disoccupazione prolungata
    eligible, reason = verifica_requisiti_rita(58, 25, 24, 50000)
    assert eligible == True, f"Dovrebbe essere eligible per R.I.T.A. (disoccupazione): {reason}"
    
    # Caso 3: Contributi insufficienti
    eligible, reason = verifica_requisiti_rita(62, 15, 0, 50000)
    assert eligible == False, "Non dovrebbe essere eligible (contributi insufficienti)"
    
    # Caso 4: Nessun montante
    eligible, reason = verifica_requisiti_rita(62, 25, 0, 0)
    assert eligible == False, "Non dovrebbe essere eligible (nessun montante)"
    
    print(f"✓ Test requisiti R.I.T.A.: OK")


def test_calcolo_incentivo_completo():
    """Test calcolo incentivo esodo completo"""
    risultato = calcola_incentivo_esodo(
        retribuzione_mensile_lorda=2500.0,
        mesi_contributi_ultimi_4_anni=48,
        regione_residenza="Lombardia",
        anni_contributi_totali=30,
        tipo_contribuzione="dipendente_privato",
        tipo_lavoro="standard",
        mesi_lavoro_prima_19=0,
        eta_lavoratore=55,
        ha_previdenza_complementare=True,
        montante_previdenza_complementare=50000.0,
        mesi_disoccupazione=0
    )
    
    # Verifica struttura risultato
    assert "dati_input" in risultato
    assert "naspi" in risultato
    assert "contributi_figurativi" in risultato
    assert "valore_tempo_libero" in risultato
    assert "incentivo_esodo" in risultato
    assert "requisiti_speciali" in risultato
    assert "previdenza_complementare" in risultato
    
    # Verifica valori chiave
    assert risultato["naspi"]["durata_mesi"] == 24
    assert risultato["naspi"]["totale_naspi_periodo"] > 0
    assert risultato["incentivo_esodo"]["incentivo_finale"] > 0
    assert risultato["incentivo_esodo"]["delta_retribuzione"] > 0
    
    # Verifica che l'incentivo sia positivo
    incentivo = risultato["incentivo_esodo"]["incentivo_finale"]
    assert incentivo > 0, f"Incentivo dovrebbe essere positivo: {incentivo}"
    
    print(f"✓ Test calcolo completo: Incentivo €{incentivo:,.2f}")
    print(f"  - Durata NASPI: {risultato['naspi']['durata_mesi']} mesi")
    print(f"  - NASPI totale: €{risultato['naspi']['totale_naspi_periodo']:,.2f}")
    print(f"  - Delta retribuzione: €{risultato['incentivo_esodo']['delta_retribuzione']:,.2f}")


def test_calcolo_con_bonus_speciale():
    """Test calcolo incentivo con bonus lavoratore precoce/usurante"""
    # Con bonus (lavoratore precoce)
    risultato_con_bonus = calcola_incentivo_esodo(
        retribuzione_mensile_lorda=2500.0,
        mesi_contributi_ultimi_4_anni=48,
        regione_residenza="Lombardia",
        anni_contributi_totali=42,
        tipo_contribuzione="dipendente_privato",
        tipo_lavoro="lavoro_notturno",
        mesi_lavoro_prima_19=12,
        eta_lavoratore=58,
        ha_previdenza_complementare=False,
        montante_previdenza_complementare=0,
        mesi_disoccupazione=0
    )
    
    # Senza bonus (standard)
    risultato_senza_bonus = calcola_incentivo_esodo(
        retribuzione_mensile_lorda=2500.0,
        mesi_contributi_ultimi_4_anni=48,
        regione_residenza="Lombardia",
        anni_contributi_totali=30,
        tipo_contribuzione="dipendente_privato",
        tipo_lavoro="standard",
        mesi_lavoro_prima_19=0,
        eta_lavoratore=55,
        ha_previdenza_complementare=False,
        montante_previdenza_complementare=0,
        mesi_disoccupazione=0
    )
    
    # Verifica che ci sia bonus
    assert risultato_con_bonus["requisiti_speciali"]["lavoratore_precoce"] == True
    assert risultato_con_bonus["requisiti_speciali"]["lavoro_usurante"] == True
    assert "bonus_speciale" in risultato_con_bonus["incentivo_esodo"]
    
    # Verifica che incentivo con bonus sia maggiore
    incentivo_con_bonus = risultato_con_bonus["incentivo_esodo"]["incentivo_finale"]
    incentivo_base_con_bonus = risultato_con_bonus["incentivo_esodo"]["incentivo_adjusted_regionale"]
    
    assert incentivo_con_bonus > incentivo_base_con_bonus, "Incentivo con bonus dovrebbe essere maggiore"
    
    print(f"✓ Test bonus speciale: OK")
    print(f"  - Con bonus: €{incentivo_con_bonus:,.2f}")
    print(f"  - Base: €{incentivo_base_con_bonus:,.2f}")


def test_variabilita_regionale():
    """Test che l'incentivo vari per regione"""
    regioni_test = ["Lombardia", "Lazio", "Campania", "Calabria"]
    incentivi = {}
    
    for regione in regioni_test:
        risultato = calcola_incentivo_esodo(
            retribuzione_mensile_lorda=2500.0,
            mesi_contributi_ultimi_4_anni=48,
            regione_residenza=regione,
            anni_contributi_totali=30,
            tipo_contribuzione="dipendente_privato",
            tipo_lavoro="standard",
            mesi_lavoro_prima_19=0,
            eta_lavoratore=55,
            ha_previdenza_complementare=False,
            montante_previdenza_complementare=0,
            mesi_disoccupazione=0
        )
        incentivi[regione] = risultato["incentivo_esodo"]["incentivo_finale"]
    
    # Verifica che ci siano differenze
    valori_unici = len(set(incentivi.values()))
    assert valori_unici > 1, "Gli incentivi dovrebbero variare per regione"
    
    # Verifica ordinamento per costo vita
    assert incentivi["Lombardia"] > incentivi["Calabria"], "Lombardia dovrebbe avere incentivo maggiore di Calabria"
    
    print(f"✓ Test variabilità regionale: OK")
    for regione, incentivo in incentivi.items():
        print(f"  - {regione}: €{incentivo:,.2f}")


def run_all_tests():
    """Esegue tutti i test"""
    print("=" * 70)
    print("ESECUZIONE TEST SUITE - INCENTIVO ESODO")
    print("=" * 70)
    print()
    
    tests = [
        ("NASPI sotto soglia", test_naspi_sotto_soglia),
        ("NASPI sopra soglia", test_naspi_sopra_soglia),
        ("NASPI tetto massimo", test_naspi_tetto_massimo),
        ("Durata NASPI", test_durata_naspi),
        ("Riduzione NASPI", test_riduzione_naspi),
        ("Contributi figurativi", test_contributi_figurativi),
        ("Costo vita regionale", test_costo_vita_regionale),
        ("Lavoratore precoce", test_lavoratore_precoce),
        ("Lavoro usurante", test_lavoro_usurante),
        ("Requisiti R.I.T.A.", test_requisiti_rita),
        ("Calcolo incentivo completo", test_calcolo_incentivo_completo),
        ("Bonus speciale", test_calcolo_con_bonus_speciale),
        ("Variabilità regionale", test_variabilita_regionale),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}:")
            print("-" * 70)
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ FALLITO: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"RISULTATI: {passed} test passati, {failed} test falliti")
    print("=" * 70)
    
    if failed == 0:
        print("✅ TUTTI I TEST SONO PASSATI!")
        return True
    else:
        print(f"❌ {failed} TEST FALLITI")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
