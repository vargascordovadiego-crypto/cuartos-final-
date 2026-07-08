"""
Genera 'clasificados_cuartos.png': una imagen limpia que muestra, sin
ambiguedad, que 4 selecciones avanzan de Cuartos de Final a Semifinales
segun el modelo (cruce armado a partir de los 8 clasificados de Octavos,
2 ya definidos en la cancha + 6 predichos por la cascada XGBoost).

Cada partido se dibuja en su propio eje (Axes) independiente, con
coordenadas locales 0..1 -- asi el texto nunca se desalinea sin importar
cuantas filas tenga la imagen final.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE = "#1a1a19"
PANEL = "#232322"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
BORDER = (1, 1, 1, 0.12)
GOOD = "#d4a339"

FIG_WIDTH_IN = 9.6
HEADER_IN = 1.75
FOOTER_IN = 0.55
CARD_IN = 1.55
CARD_GAP_FRAC = 0.14  # fraccion de CARD_IN dejada como separacion entre tarjetas


def _dibujar_tarjeta(fig, bottom_frac, height_frac, fila):
    ax = fig.add_axes((0.03, bottom_frac, 0.94, height_frac))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    local, visitante = fila["equipo_local"], fila["equipo_visitante"]
    gano_local = fila["ganador_predicho"] == local
    ganador = fila["ganador_predicho"]
    perdedor = visitante if gano_local else local

    caja = FancyBboxPatch((0.0, 0.06), 1.0, 0.88,
                           boxstyle="round,pad=0,rounding_size=0.10",
                           linewidth=1.1, edgecolor=BORDER, facecolor=PANEL,
                           transform=ax.transAxes, zorder=1)
    ax.add_patch(caja)

    ax.text(0.055, 0.62, "✓", transform=ax.transAxes, fontsize=16,
            color=GOOD, va="center", ha="center", fontweight="bold", zorder=3)
    ax.text(0.10, 0.62, ganador, transform=ax.transAxes, fontsize=17,
            fontweight="bold", color=INK_PRIMARY, va="center", ha="left", zorder=3)
    ax.text(0.10, 0.24, f"vence a  {perdedor}", transform=ax.transAxes, fontsize=10.5,
            color=INK_MUTED, va="center", ha="left", zorder=3)

    ax.text(0.63, 0.44, fila["marcador_probable"], transform=ax.transAxes,
            fontsize=15, color=INK_SECONDARY, va="center", ha="center",
            family="monospace", zorder=3)

    if fila["ya_jugado"]:
        etiqueta, color_badge = "RESULTADO REAL", INK_MUTED
        detalle = ""
    else:
        etiqueta, color_badge = "PREDICCION IA", GOOD
        detalle = f"{fila['confianza']*100:.0f}% confianza (Monte Carlo)"

    ax.text(0.955, 0.68, etiqueta, transform=ax.transAxes, fontsize=8.7,
            color=color_badge, va="center", ha="right", fontweight="bold",
            family="monospace", zorder=3)
    if detalle:
        ax.text(0.955, 0.40, detalle, transform=ax.transAxes, fontsize=9.3,
                color=INK_SECONDARY, va="center", ha="right", zorder=3)


def generar_imagen(resultados, ruta_salida, fecha_corte="2026-07-07"):
    resultados = resultados.sort_values("match_id").reset_index(drop=True)
    n = len(resultados)

    fig_h = HEADER_IN + FOOTER_IN + n * CARD_IN
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    fig.text(0.5, 1 - (0.30 / fig_h), "COPA MUNDIAL 2026", fontsize=13,
              color=INK_MUTED, ha="center", va="top", fontweight="bold", family="monospace")
    fig.text(0.5, 1 - (0.62 / fig_h), "Clasificados de Cuartos de Final a Semifinales",
              fontsize=20, color=INK_PRIMARY, ha="center", va="top", fontweight="bold")
    fig.text(0.5, 1 - (1.02 / fig_h),
              f"Proyeccion generada por la cascada XGBoost del modelo  ·  corte de datos {fecha_corte}",
              fontsize=10, color=INK_SECONDARY, ha="center", va="top")

    card_slot_frac = CARD_IN / fig_h
    card_height_frac = card_slot_frac * (1 - CARD_GAP_FRAC)
    gap_frac = card_slot_frac * CARD_GAP_FRAC

    top_y_in = fig_h - HEADER_IN
    for i, fila in resultados.iterrows():
        top_frac = top_y_in / fig_h - i * card_slot_frac
        bottom_frac = top_frac - card_height_frac
        _dibujar_tarjeta(fig, bottom_frac, card_height_frac, fila)

    fig.text(0.5, (0.22 / fig_h),
              "Metodologia: cascada XGBoost (regresion Tweedie de goles + clasificador 1X2 calibrado) · "
              "truco del espejo para sede neutral · simulacion Monte Carlo de penaltis en empates",
              fontsize=7.6, color=INK_MUTED, ha="center", va="bottom")

    fig.savefig(ruta_salida, facecolor=SURFACE)
    plt.close(fig)
