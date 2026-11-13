# -*- coding: utf-8 -*-
"""
===============================================================================
 MODELO DE CAJA (BOX MODEL) EN ESTADO NO ESTACIONARIO
 Simulacion de concentraciones de contaminantes en Bogota
===============================================================================

 Basado en la presentacion "Introduccion a la modelacion de la calidad del aire
 - Modelo de caja" (Prof. Nestor Y. Rojas, Universidad Nacional de Colombia).

 Ecuacion diferencial (balance de masa en la caja):

     dC/dt + (u/L - k1) C - (b*u/L + Q/(W*L*H)) = 0

 Solucion analitica no estacionaria (se integra hora a hora, encadenando el
 valor final de una hora como condicion inicial de la siguiente, porque u, H y
 Q cambian cada hora):

     C(t) = [ (Q/(W*L*H) + b*u/L) / (u/L - k1) ] * (1 - exp(-(u/L - k1) t))
            + C0 * exp(-(u/L - k1) t)

 Donde, por cada hora:
   C   = concentracion dentro de la caja (resultado)
   C0  = concentracion al inicio de la hora (C de la hora anterior)
   Q   = emision total (g/h)   -> de los inventarios (CMB + FF + RDST)
   b   = concentracion de fondo -> estacion Guaymaral (minimo observado)
   u   = velocidad del viento (m/s) -> estacion CAR (respaldo Guaymaral)
   H   = altura de la capa de mezcla (m) -> ERA5 (CapaDeMezcla.csv)
   L,W = dimensiones de la caja (m)
   k1  = constante de reaccion de primer orden (0 para no reactivos como CO)

 La estacion Centro de Alto Rendimiento (CAR) se usa para VALIDAR el modelo.

===============================================================================
 NOTA: La descarga de la capa de mezcla (ERA5) se hace con el script aparte
 'get_era5.py'. Este script del modelo solo LEE el archivo CapaDeMezcla.csv.
===============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla (guarda figuras a archivo)
import matplotlib.pyplot as plt

# Carpeta donde reside este script. Todos los archivos de entrada/salida se
# resuelven respecto a esta carpeta, de modo que el script funcione sin importar
# desde que directorio se ejecute (o con doble clic).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
#  >>>>>>>>>>>>>>>>>>>>>>  CONFIGURACION (EDITAR AQUI)  <<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================

# ---- Contaminante a simular ----
# Opciones tipicas segun inventario y estaciones: "CO", "PM25", "NOX"
# (Para empezar y validar con el ejemplo del profesor, usar "CO").
CONTAMINANTE = "CO"

# ---- Ventana temporal de la simulacion ----
FECHA_INICIO = "2023-03-13"   # Lunes 00:00 (hora local COT). Editable.
N_DIAS       = 7              # 7 dias = 168 horas

# ---- Dimensiones de la caja (Bogota, segun el profesor) ----
L_KM = 20.0   # largo de la caja [km]
W_KM = 40.0   # ancho de la caja [km]

# ---- Cinetica de reaccion de primer orden ----
# k1 = 0  -> contaminante NO reactivo (CO, PM). Mantener en 0 salvo reactivos.
# Para reactivos (p.ej. NOX) se puede asignar un k1 efectivo [1/h].
K1_PER_H = 0.0

# ---- Zona horaria de la capa de mezcla (ERA5) ----
# ERA5 viene en UTC; las estaciones estan en hora local COT = UTC-5.
ERA5_UTC_OFFSET_H = -5

# ---- Manejo de huecos ----
# "NaN": NO interpolar; las horas con datos faltantes quedan sin resultado.
MANEJO_HUECOS = "NaN"

# ---- Concentracion de fondo b ----
# "min_global": un unico valor = minimo observado en Guaymaral en la ventana.
#               (es lo que pide el enunciado: "concentraciones minimas observadas")
# "serie":      usar la serie horaria de Guaymaral tal cual (sin rellenar huecos).
MODO_FONDO = "min_global"

# ---- Fuente del viento u ----
# "CAR": usar viento de la estacion CAR (mas completo en esta semana).
# "GUAYMARAL": usar viento de Guaymaral.
FUENTE_VIENTO = "CAR"

# ---- Escenario de politica publica (cambio en emisiones) ----
# Reduccion fraccional de la emision Q de ciertas categorias de fuente.
# Ejemplo: 0.30 = reducir 30% las emisiones de las categorias que empiecen
# por alguno de los prefijos listados (p.ej. buses/camiones diesel).
APLICAR_ESCENARIO   = True
ESCENARIO_NOMBRE    = "Reduccion 30% fuentes diesel de carga/buses"
ESCENARIO_REDUCCION = 0.30
# Prefijos de categoria a afectar (segun codigos del inventario, p.ej. "BS_DSEL").
# Si la lista esta vacia, se aplica la reduccion a TODA la emision.
ESCENARIO_PREFIJOS  = ["BS_DSEL", "BA_DSEL", "AT_DSEL"]

# ---- Archivos de entrada (se resuelven respecto a la carpeta del script) ----
ARCH_INV_CMB  = os.path.join(BASE_DIR, "Summary of CMB_2023m3_adj2.nc.xlsx")
ARCH_INV_FF   = os.path.join(BASE_DIR, "Summary of FF_2023v2.nc.xlsx")
ARCH_INV_RDST = os.path.join(BASE_DIR, "Summary of RDST_2023m3_adj2.nc.xlsx")
ARCH_ERA5     = os.path.join(BASE_DIR, "CapaDeMezcla.csv")
ARCH_EST_CAR  = os.path.join(BASE_DIR, "StationsReport_20261012223526.xlsx")  # Centro de Alto Rendimiento
ARCH_EST_GUA  = os.path.join(BASE_DIR, "StationsReport_20261012223539.xlsx")  # Guaymaral

# ---- Archivos de salida ----
DIR_SALIDA    = os.path.join(BASE_DIR, "resultados")

# =============================================================================
#  Mapeo de nombres: cada contaminante -> como se llama en inventario y estacion
# =============================================================================
# hoja_inv:   nombre base de la hoja "<X>_profile" en los inventarios
# col_est:    nombre de la columna en los reportes de estacion (CAR / Guaymaral)
# unidad_est: unidad de la estacion (solo informativo)
# pm_micro:   True si la estacion lo reporta en ug/m3 (PM); False si ppm/ppb (gases)
MAPEO = {
    "CO":   {"hoja_inv": "CO",   "col_car": "CO",    "col_gua": "CO",
             "unidad_est": "ppm", "es_pm": False, "peso_mol": 28.01},
    "PM25": {"hoja_inv": "PM25", "col_car": "PM2.5", "col_gua": "PM2.5",
             "unidad_est": "ug/m3", "es_pm": True, "peso_mol": None},
    "PM10": {"hoja_inv": "PM10", "col_car": "PM10",  "col_gua": "PM10",
             "unidad_est": "ug/m3", "es_pm": True, "peso_mol": None},
    "NOX":  {"hoja_inv": "NOx",  "col_car": "NOX",   "col_gua": "NOX",
             "unidad_est": "ppb", "es_pm": False, "peso_mol": 46.01},
}

# Constantes fisicas / de conversion
R_GAS   = 8.314462        # J/(mol K)
T_REF_K = 298.15          # K  (condiciones de referencia para ppm<->ug/m3)
P_REF   = 101325.0        # Pa


# =============================================================================
#  UTILIDADES
# =============================================================================
def parse_fecha_estacion(s):
    """Convierte 'dd-mm-YYYY HH:MM' a Timestamp. Maneja la convencion 24:00
    (medianoche escrita como hora 24 del dia) -> 00:00 del dia siguiente."""
    s = str(s).strip()
    if not re.match(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2}", s):
        return pd.NaT
    fecha, hora = s.split(" ")
    hh, mm = hora.split(":")
    base = pd.to_datetime(fecha, format="%d-%m-%Y")
    return base + pd.Timedelta(hours=int(hh), minutes=int(mm))


def ventana_horaria():
    """Construye el indice horario de la ventana (168 h por defecto)."""
    t0 = pd.Timestamp(FECHA_INICIO + " 00:00")
    n = N_DIAS * 24
    return pd.date_range(t0, periods=n, freq="h")


def tipo_dia(ts):
    """Mapea un Timestamp al tipo de dia usado en el inventario."""
    wd = ts.weekday()  # 0=lun ... 6=dom
    if wd <= 4:
        return "lun-vie"
    elif wd == 5:
        return "sab"
    else:
        return "dom"


# =============================================================================
#  1) LECTURA DE INVENTARIOS -> Q(t) horaria por tipo de dia
# =============================================================================
def leer_perfil_inventario(archivo, hoja_base):
    """Lee una hoja '<X>_profile' y devuelve un DataFrame largo con columnas:
    ['wdy','Category', 0..23] y los valores de emision (kg/h) por hora.
    Devuelve None si la hoja no existe (p.ej. CO no esta en RDST)."""
    xl = pd.ExcelFile(archivo)
    nombre = None
    for s in xl.sheet_names:
        if s.upper().replace("_PROFILE", "") == hoja_base.upper():
            nombre = s
            break
    if nombre is None:
        return None

    raw = pd.read_excel(archivo, sheet_name=nombre, header=None)
    # La fila 11 (0-indexada) contiene el encabezado: wdy, Category, 0,1,...,23
    # Estructura por POSICION: col0=wdy, col1=Category, col2..col25 = horas 0..23
    datos = raw.iloc[12:].copy()
    datos = datos[datos.iloc[:, 1].notna()]  # Category no vacia

    registros = []
    for _, fila in datos.iterrows():
        wdy = fila.iloc[0]
        cat = fila.iloc[1]
        horas = pd.to_numeric(fila.iloc[2:26], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(horas) < 24:
            horas = np.pad(horas, (0, 24 - len(horas)))
        registros.append((wdy, cat, horas))

    out = pd.DataFrame({
        "wdy": [r[0] for r in registros],
        "Category": [str(r[1]) for r in registros],
    })
    for h in range(24):
        out[h] = [r[2][h] for r in registros]
    return out


def construir_Q_por_tipo_dia(contaminante, aplicar_escenario=False):
    """Suma todos los inventarios y categorias -> emision total por tipo de dia
    y por hora, en g/h. Devuelve dict: {tipo_dia: np.array(24) en g/h}.
    Si aplicar_escenario=True, reduce las categorias indicadas."""
    hoja = MAPEO[contaminante]["hoja_inv"]
    fuentes = [ARCH_INV_CMB, ARCH_INV_FF, ARCH_INV_RDST]

    acumulado = {}  # {tipo_dia: array(24)}
    detalle_categorias = {}  # para reporte del escenario

    for arch in fuentes:
        perfil = leer_perfil_inventario(arch, hoja)
        if perfil is None:
            continue
        for td in ["lun-vie", "sab", "dom"]:
            sub = perfil[perfil["wdy"] == td]
            if sub.empty:
                continue
            for _, fila in sub.iterrows():
                cat = str(fila["Category"])
                vals = np.array([float(fila[i]) for i in range(24)])  # kg/h

                # --- escenario de politica: reducir ciertas categorias ---
                if aplicar_escenario:
                    if (not ESCENARIO_PREFIJOS) or any(
                        cat.startswith(p) for p in ESCENARIO_PREFIJOS
                    ):
                        vals = vals * (1.0 - ESCENARIO_REDUCCION)

                acumulado.setdefault(td, np.zeros(24))
                acumulado[td] += vals
                detalle_categorias.setdefault(cat, 0.0)
                detalle_categorias[cat] += float(np.nansum(vals))

    # kg/h -> g/h
    Q = {td: arr * 1000.0 for td, arr in acumulado.items()}
    return Q, detalle_categorias


def serie_Q_ventana(Q_tipo_dia, indice):
    """Expande Q(tipo_dia, hora) a la serie horaria de la ventana (g/h)."""
    out = np.full(len(indice), np.nan)
    for i, ts in enumerate(indice):
        td = tipo_dia(ts)
        if td in Q_tipo_dia:
            out[i] = Q_tipo_dia[td][ts.hour]
    return out


# =============================================================================
#  2) LECTURA DE CAPA DE MEZCLA (ERA5) -> H(t) en hora local
# =============================================================================
def leer_capa_mezcla(indice):
    df = pd.read_csv(ARCH_ERA5, parse_dates=["valid_time"])
    df["t_local"] = df["valid_time"] + pd.Timedelta(hours=ERA5_UTC_OFFSET_H)
    df = df.set_index("t_local").sort_index()
    # columna de altura: 'blh' (m). 'ZI' es la version redondeada con -999 = faltante.
    serie = df["blh"].reindex(indice)
    return serie.to_numpy(dtype=float)


# =============================================================================
#  3) LECTURA DE ESTACIONES (CAR y Guaymaral)
# =============================================================================
def leer_estacion(archivo):
    raw = pd.read_excel(archivo, sheet_name="Sheet1", header=None)
    variables = [str(v) for v in raw.iloc[3].tolist()]
    cols = ["DateTime"] + variables[1:]
    d = raw.iloc[5:].copy()
    d.columns = cols
    d = d.replace("----", np.nan)
    d["DateTime"] = d["DateTime"].apply(parse_fecha_estacion)
    d = d.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
    # convertir a numerico todo lo posible
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def serie_columna(df_est, nombre_col, indice):
    """Extrae una columna de la estacion alineada a la ventana (NaN donde falte)."""
    if nombre_col not in df_est.columns:
        return np.full(len(indice), np.nan)
    return df_est[nombre_col].reindex(indice).to_numpy(dtype=float)


# =============================================================================
#  4) MOTOR DEL MODELO DE CAJA (no estacionario, encadenado hora a hora)
# =============================================================================
def paso_box_model(C0, Q_gh, b, u, H, L, W, k1, dt_h=1.0):
    """Integra UNA hora con la solucion analitica.
       Unidades coherentes: concentracion en g/m3.
       Q [g/h], b [g/m3], u [m/s], H,L,W [m], k1 [1/h].
       Devuelve C al final de la hora (g/m3) o NaN si faltan datos."""
    if np.isnan(Q_gh) or np.isnan(b) or np.isnan(u) or np.isnan(H) or H <= 0:
        return np.nan

    u_h = u * 3600.0            # m/s -> m/h  (para trabajar en horas)
    a = u_h / L - k1            # coeficiente del termino lineal [1/h]
    # termino fuente (independiente): Q/(W L H) + b u / L   en g/(m3 h)
    fuente = Q_gh / (W * L * H) + b * u_h / L

    if abs(a) < 1e-12:
        # limite a->0: C = C0 + fuente*dt
        return C0 + fuente * dt_h

    C_est = fuente / a                      # concentracion de estado estacionario
    return C_est * (1.0 - np.exp(-a * dt_h)) + C0 * np.exp(-a * dt_h)


def correr_modelo(indice, Qserie, b_serie, u_serie, H_serie, C0_inicial,
                  L, W, k1):
    """Corre la simulacion encadenada en toda la ventana. Devuelve array de C
    en las unidades de 'b' (g/m3)."""
    n = len(indice)
    C = np.full(n, np.nan)
    C_prev = C0_inicial
    for i in range(n):
        Ci = paso_box_model(C_prev, Qserie[i], b_serie[i], u_serie[i],
                            H_serie[i], L, W, k1)
        C[i] = Ci
        # Encadenar: si esta hora dio NaN (dato faltante), conservar el ultimo
        # valido SOLO como condicion inicial interna (no se reporta como dato),
        # para no cortar toda la simulacion. El resultado reportado sigue NaN.
        if not np.isnan(Ci):
            C_prev = Ci
    return C


# =============================================================================
#  5) CONVERSIONES DE UNIDADES
# =============================================================================
def ppm_a_gm3(c_ppm, peso_mol, T=T_REF_K, P=P_REF):
    """ppm (volumen) -> g/m3 para un gas."""
    # n/V = P/(R T) mol/m3 ; c_ppm * 1e-6 = fraccion molar
    mol_por_m3 = P / (R_GAS * T)
    return c_ppm * 1e-6 * mol_por_m3 * peso_mol  # g/m3


def gm3_a_ppm(c_gm3, peso_mol, T=T_REF_K, P=P_REF):
    mol_por_m3 = P / (R_GAS * T)
    return c_gm3 / (mol_por_m3 * peso_mol) / 1e-6


def ugm3_a_gm3(c_ug):
    return c_ug * 1e-6


def gm3_a_ugm3(c_g):
    return c_g * 1e6


# =============================================================================
#  6) METRICAS DE DESEMPENO
# =============================================================================
def metricas(sim, obs):
    """Calcula metricas usando solo horas donde AMBOS tienen dato (sin rellenar)."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = ~np.isnan(sim) & ~np.isnan(obs)
    n = int(mask.sum())
    if n < 2:
        return {"n": n, "RMSE": np.nan, "MBE": np.nan, "MAE": np.nan, "r": np.nan}
    s, o = sim[mask], obs[mask]
    rmse = float(np.sqrt(np.mean((s - o) ** 2)))
    mbe = float(np.mean(s - o))
    mae = float(np.mean(np.abs(s - o)))
    r = float(np.corrcoef(s, o)[0, 1]) if np.std(s) > 0 and np.std(o) > 0 else np.nan
    return {"n": n, "RMSE": rmse, "MBE": mbe, "MAE": mae, "r": r}


# =============================================================================
#  PROGRAMA PRINCIPAL
# =============================================================================
def main():
    os.makedirs(DIR_SALIDA, exist_ok=True)
    cont = CONTAMINANTE.upper()
    if cont not in MAPEO:
        raise ValueError(f"Contaminante '{cont}' no esta en el MAPEO. "
                         f"Opciones: {list(MAPEO.keys())}")
    info = MAPEO[cont]
    L = L_KM * 1000.0
    W = W_KM * 1000.0
    k1 = K1_PER_H

    print("=" * 75)
    print(f" MODELO DE CAJA NO ESTACIONARIO  |  Contaminante: {cont}")
    print(f" Ventana: {FECHA_INICIO} 00:00  +{N_DIAS} dias ({N_DIAS*24} h)  "
          f"|  Caja: {L_KM} x {W_KM} km  |  k1 = {k1} 1/h")
    print("=" * 75)

    idx = ventana_horaria()

    # --- 1) Emisiones Q(t) [g/h] (caso base) ---
    Q_td, _ = construir_Q_por_tipo_dia(cont, aplicar_escenario=False)
    if not Q_td:
        raise RuntimeError(f"No se encontraron emisiones de {cont} en los "
                          f"inventarios (revisar nombre de hoja).")
    Qserie = serie_Q_ventana(Q_td, idx)
    print(f" Emisiones {cont}: Q medio = {np.nanmean(Qserie):.3e} g/h "
          f"({np.nanmean(Qserie)/1e6:.2f} ton/h)")

    # --- 2) Capa de mezcla H(t) [m] ---
    H_serie = leer_capa_mezcla(idx)
    print(f" Capa de mezcla H: {np.nanmin(H_serie):.0f} - {np.nanmax(H_serie):.0f} m "
          f"(faltan {np.isnan(H_serie).sum()} h)")

    # --- 3) Estaciones ---
    car = leer_estacion(ARCH_EST_CAR)
    gua = leer_estacion(ARCH_EST_GUA)

    # viento u(t) [m/s]
    if FUENTE_VIENTO == "CAR":
        u_serie = serie_columna(car, "Vel Viento", idx)
    else:
        u_serie = serie_columna(gua, "Vel Viento 10M", idx)
    print(f" Viento u ({FUENTE_VIENTO}): medio = {np.nanmean(u_serie):.2f} m/s "
          f"(faltan {np.isnan(u_serie).sum()} h)")

    # observado para validacion (CAR)
    obs_est = serie_columna(car, info["col_car"], idx)      # en unidades de estacion
    # fondo (Guaymaral)
    fondo_est = serie_columna(gua, info["col_gua"], idx)    # en unidades de estacion

    # --- 4) Pasar todo a g/m3 para el modelo ---
    if info["es_pm"]:
        obs_gm3 = ugm3_a_gm3(obs_est)
        fondo_gm3 = ugm3_a_gm3(fondo_est)
        to_user = gm3_a_ugm3          # para mostrar resultados
        unidad = "ug/m3"
    else:
        pm = info["peso_mol"]
        obs_gm3 = ppm_a_gm3(obs_est, pm) if info["unidad_est"] == "ppm" \
            else ppm_a_gm3(obs_est * 1e-3, pm)  # ppb -> ppm
        fondo_gm3 = ppm_a_gm3(fondo_est, pm) if info["unidad_est"] == "ppm" \
            else ppm_a_gm3(fondo_est * 1e-3, pm)
        to_user = (lambda x: gm3_a_ppm(x, pm)) if info["unidad_est"] == "ppm" \
            else (lambda x: gm3_a_ppm(x, pm) * 1e3)
        unidad = info["unidad_est"]

    # fondo b: valor unico (minimo) o serie
    if MODO_FONDO == "min_global":
        b_val = np.nanmin(fondo_gm3)
        b_serie = np.full(len(idx), b_val)
        print(f" Fondo b (min Guaymaral): {to_user(b_val):.3f} {unidad}")
    else:
        b_serie = fondo_gm3
        print(f" Fondo b (serie Guaymaral): faltan {np.isnan(b_serie).sum()} h")

    # condicion inicial C0: primer observado disponible o el fondo
    primeros_obs = obs_gm3[~np.isnan(obs_gm3)]
    C0_ini = primeros_obs[0] if primeros_obs.size else np.nanmin(fondo_gm3)

    # --- 5) Correr modelo (caso base) ---
    C_base_gm3 = correr_modelo(idx, Qserie, b_serie, u_serie, H_serie,
                              C0_ini, L, W, k1)
    sim_base = to_user(C_base_gm3)
    obs_user = to_user(obs_gm3)

    met = metricas(sim_base, obs_user)
    print("-" * 75)
    print(f" DESEMPENO (simulado vs observado CAR) [{unidad}]:")
    print(f"   n = {met['n']}  |  RMSE = {met['RMSE']:.3f}  |  MBE = {met['MBE']:.3f}"
          f"  |  MAE = {met['MAE']:.3f}  |  r = {met['r']:.3f}")

    # --- 6) Escenario de politica publica ---
    sim_esc = None
    if APLICAR_ESCENARIO:
        Q_td_e, _ = construir_Q_por_tipo_dia(cont, aplicar_escenario=True)
        Qserie_e = serie_Q_ventana(Q_td_e, idx)
        C_esc_gm3 = correr_modelo(idx, Qserie_e, b_serie, u_serie, H_serie,
                                  C0_ini, L, W, k1)
        sim_esc = to_user(C_esc_gm3)
        red_media = np.nanmean(sim_base - sim_esc)
        red_pct = 100.0 * np.nanmean((sim_base - sim_esc) /
                                     np.where(sim_base == 0, np.nan, sim_base))
        print("-" * 75)
        print(f" ESCENARIO: {ESCENARIO_NOMBRE}")
        print(f"   Reduccion media de concentracion: {red_media:.3f} {unidad} "
              f"({red_pct:.1f} %)")

    # --- 7) Exportar CSV de resultados ---
    salida = pd.DataFrame({
        "fecha_hora": idx,
        "tipo_dia": [tipo_dia(t) for t in idx],
        "Q_g_h": Qserie,
        "H_m": H_serie,
        "u_m_s": u_serie,
        f"fondo_b_{unidad}": to_user(b_serie),
        f"sim_base_{unidad}": sim_base,
        f"obs_CAR_{unidad}": obs_user,
    })
    if sim_esc is not None:
        salida[f"sim_escenario_{unidad}"] = sim_esc
    csv_path = os.path.join(DIR_SALIDA, f"resultados_{cont}.csv")
    salida.to_csv(csv_path, index=False)
    print("-" * 75)
    print(f" CSV de resultados: {csv_path}")

    # --- 8) Graficas ---
    # (a) Simulado vs observado
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(idx, obs_user, color="black", lw=1.6, label=f"Observado CAR")
    ax.plot(idx, sim_base, color="tab:red", lw=1.6, label="Simulado (base)")
    if sim_esc is not None:
        ax.plot(idx, sim_esc, color="tab:green", lw=1.4, ls="--",
                label="Simulado (escenario)")
    ax.set_title(f"Modelo de caja - {cont}  |  {FECHA_INICIO} (semana)")
    ax.set_ylabel(f"Concentracion [{unidad}]")
    ax.set_xlabel("Fecha / hora (COT)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig1 = os.path.join(DIR_SALIDA, f"serie_{cont}.png")
    fig.savefig(fig1, dpi=130)
    plt.close(fig)

    # (b) Dispersion simulado vs observado
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    m = ~np.isnan(sim_base) & ~np.isnan(obs_user)
    ax.scatter(obs_user[m], sim_base[m], s=18, alpha=0.6, color="tab:blue")
    lim = [0, np.nanmax([np.nanmax(obs_user[m]) if m.any() else 1,
                         np.nanmax(sim_base[m]) if m.any() else 1])]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel(f"Observado [{unidad}]")
    ax.set_ylabel(f"Simulado [{unidad}]")
    ax.set_title(f"Dispersion {cont} (r = {met['r']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig2 = os.path.join(DIR_SALIDA, f"dispersion_{cont}.png")
    fig.savefig(fig2, dpi=130)
    plt.close(fig)

    print(f" Figuras: {fig1}  |  {fig2}")
    print("=" * 75)
    print(" Listo.")


if __name__ == "__main__":
    main()
