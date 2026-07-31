# Respuestas a las observaciones del seminario — RL-RP-SMS-EMOA

Documento de acompañamiento a los cambios de código. Cada sección responde una
observación y apunta al archivo donde quedó implementada.

---

## 1. ¿Cómo se manejará el valor de diversidad?

**Problema encontrado.** Dos de las cinco variables de estado se normalizaban con
un **máximo corriente**: `D_norm(t)=D(t)/max_{t'≤t}D(t')`, `HV_norm(t)=HV(t)/max_{t'≤t}HV(t')`.
Un máximo corriente es un estadístico de la *trayectoria*, no de la población: el
mismo frente podía ser el estado 137 en t=1000 y el 42 en t=50000. Eso vuelve el
proceso de decisión **no estacionario**, y Q-learning tabular sólo tiene garantías
sobre un MDP estacionario.

**Corrección** (`indicators.py`, `state.py`). Cotas absolutas:

- `D_norm = D(Â)/½ ∈ [0,1]` — en `[0,1]^m` la desviación estándar media por
  objetivo está acotada por ½ (máximo en la distribución degenerada mitad-en-0 /
  mitad-en-1).
- `HV_norm = HV(Â,z_ref)/∏_i z_ref,i ∈ [0,1]` — normalizado por el volumen de la
  caja, no por el mejor HV histórico.

Además se separan explícitamente las dos nociones que estaban mezcladas:
**dispersión/extensión** (`mean_dispersion`, `extent`) y **uniformidad**
(energía de Riesz). Ambas se registran por separado en cada generación.

---

## 2. ¿Está normalizado el espacio objetivo?

Sí, pero **el nadir estaba mal estimado**: `estimate_ideal_nadir` tomaba el máximo
sobre **toda la población**, no sobre el frente no dominado, de modo que los peores
individuos dominados dictaban la escala. Corregido a máximo componente a componente
sobre el frente no dominado (`indicators.estimate_ideal_nadir`, `from_front=True`).

Con eso los pasos (0.01, 1/H, 1.0) sí son desplazamientos en unidades normalizadas:
1% de la extensión del frente, el espaciado clásico de Ishibuchi con H=12, y un
rango nadir–ideal completo.

---

## 3. Tres objetivos. ¿Es escalable?

| Componente | Escalamiento en m |
|---|---|
| Espacio de acciones | 2m (lineal) |
| Espacio de estados | constante |
| HV exacto (WFG) | **exponencial** |
| `hv_contributions` | O(n) llamadas a HV por generación |

Se añadió un backend Monte Carlo (`hypervolume(..., backend="mc")`, automático
para m>5), así que el algoritmo ya no depende de que HV exacto sea costeable.

**Pero el límite real no es m, es la complejidad muestral** — ver sección 8.
Se añadió también R2-EMOA como control: su selección cuesta O(|W|·|A|·m) y no
explota con m (sección 10).

---

## 4. ¿Cómo se detecta la geometría en el estado?

**Antes no se detectaba.** `ĝ` medía la razón de contribuciones al HV entre la
mitad exterior e interior del frente: eso es *dónde está la presión de selección*,
no *qué forma tiene el frente*. Peor, **depende de z_ref**, que es justo lo que el
planeador mueve, así que confundía causa con efecto.

**Corrección** (`indicators.estimate_curvature_p`, `geometry_gamma`). Se ajusta el
exponente de curvatura *p* del modelo `Σ_i f̂_i^p = 1` por bisección geométrica
sobre la mediana, y se usa

> **γ = p/(1+p) ∈ (0,1)**  —  γ<½ convexo, γ=½ lineal, γ>½ cóncavo

acotado por construcción, sin normalización corriente. Valores medidos sobre los
frentes analíticos reales (m=3):

| DTLZ1 | DTLZ2 | DTLZ7 | WFG4 | WFG9 | −DTLZ1 | −DTLZ2 | IMOP4 | IMOP5 | IMOP6 | IMOP7 | IMOP8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 2.00 | 2.45 | 2.01 | 2.00 | 4.08 | 2.58 | 1.59 | 1.86 | 1.73 | 2.00 | 1.62 |

Se añadió además **ι (invertidez)** = percentil bajo de `Σ_i f̂_i` menos 1,
normalizado: distingue frentes regulares (ι≈0) de invertidos (Minus-DTLZ1 ι≈0.50).
Se registra como diagnóstico; incorporarlo al estado multiplicaría |S|, lo cual es
contraproducente dada la sección 8.

---

## 5. ¿Hasta dónde puede llegar el movimiento? ¿Está acotado?

**No lo estaba por arriba.** La actualización era `z_i ← max(1, z_i + s·δ)`. Con el
régimen grueso (δ=1) el punto podía irse a 10³ o más, donde el paisaje de HV es
plano y SMS-EMOA degenera en un método que empuja la población a los extremos.

**Corrección** (`rl_planner.apply_action`), siguiendo la sugerencia recibida:

```
z_i(t+1) = clip( z_i(t) + s·δ_k , 1+ε , z_max )    ε = 1e-3,  z_max = 10
```

La cota inferior es estrictamente **mayor** que 1 para que las soluciones extremas
conserven contribución positiva al HV (con z_i=1 exacto, una solución que alcanza
el nadir en el objetivo i contribuye volumen cero y sería la primera eliminada).

*Alcanzabilidad:* desde z=1 el régimen grueso llega a z_max en 9 pasos, el
balanceado en 108, el fino en 900 — despreciable frente a ρ·T_max ≈ 7×10⁴. La caja
restringe **dónde puede terminar**, no **qué puede explorar**.

*Evidencia empírica:* sin la cota, z_ref satura. En corridas medidas terminó en
(7.0, 9.0, 9.0) y (3.9, 1.1, 4.6) — pegado a la frontera. Causa: con Q sin entrenar
y σ₀=(0.10,0.10,0.80), el punto hace una **caminata aleatoria** con pasos ±1.

---

## 6. IGD+ — ¿qué conjunto de referencia?

Se encontraron **dos bugs graves**.

**(a) Fallback de muestreo aleatorio.** Para WFG e IMOP, Z se generaba muestreando
el espacio de decisión al azar y filtrando no dominados. Eso **no es una muestra
del frente de Pareto**: muestreo uniforme de un WFG de 14 variables prácticamente
nunca alcanza g=0. Los valores "IGD+" reportados no eran IGD+.

**(b) El frente de Minus-DTLZ estaba mal.** El código hacía `PF_minus = −PF_base`.
Como `f = (1+g)·h(x_pos)`, minimizar −f exige g en su **máximo**, no en 0:

> **PF(Minus-DTLZ) = −(1+g_max) · PF(DTLZ)**

Factores verificados numéricamente: **Minus-DTLZ2 → 3.5**, **Minus-DTLZ1 → 1102.30**.
El conjunto de referencia estaba equivocado por un factor de 3.5 y por *tres órdenes
de magnitud* respectivamente; por eso IGD+ salía exactamente 0.0 (la aproximación
dominaba todo Z). Se confirmó que los puntos construidos en g_max no son dominados
por 2×10⁴ muestras aleatorias.

**Ahora Z es siempre analítico** (`problems.reference_set`): pymoo con retícula
Das-Dennis para DTLZ/WFG, negación escalada para Minus-DTLZ, y el `GetOptimum`
original de PlatEMO para IMOP. Validación (`tests_reference_sets.py`):

```
problema        m    |Z|  Z dominados  d_rel medio  d_rel p95
dtlz1           3   1953            0      0.00471    0.00693
dtlz2           3   1953            0      0.00632    0.01048
minus-dtlz1     3   1953            0      0.00471    0.00693
minus-dtlz2     3   1953            0      0.00632    0.01048
wfg4            3   1953            0      0.00536    0.00992
wfg9            3   1953            0      0.00602    0.01088
imop1..imop8  2/3  ~2000            0     <0.0066     <0.0264
```

Cero puntos dominados y distancia a la variedad óptima real por debajo del 1% del
diámetro (salvo IMOP8, 2.6%, el multimodal extremo).

> **|Z| debe reportarse siempre**: IGD+ es un estadístico muestral de Z y valores
> con distinto |Z| no son comparables.

---

## 7. ¿Cómo influye la energía de Riesz en la decisión?

Entra por **tres vías**, y había un bug en la primera:

1. **Estado.** `E_norm` = razón de energías logarítmicas consecutivas. El código
   hacía `min(ratio,1.0)` — **el recorte hacía invisible el deterioro de
   uniformidad**; el planeador nunca podía ver que un movimiento había empeorado
   la distribución. Ahora `E_norm = clip(ratio,0,2)/2`, donde ½ = sin cambio.
2. **Recompensa.** `r_t = ΔHV + α·(E_norm(t) − E_norm(t+1))`.
3. **Pagos de la dinámica replicadora.** `u_k = β(D_norm)·EMA[ΔU]_k + (1−β)·EMA[ΔHV]_k`.
   En el código original esta variable se llamaba `delta_D` (dispersión) pero
   contenía una diferencia de **energía**; renombrada a `delta_U` y documentada.

La energía pasó a ser **promedio por par** (dividida entre n(n−1)); la suma cruda
escala como n², así que un cambio en |A| movía la variable de estado aunque la
geometría no cambiara.

**Cuestión de diseño abierta.** La recompensa usaba HV contra el z_ref *móvil*:
como `HV(A,z_ref)/vol(z_ref)` cambia al mover z_ref aunque la población esté
intacta, el planeador se premiaba en parte por **mover la portería**. Se implementó
`reward_hv="fixed"` (por defecto) y `"adaptive"` (el original) para compararlos.

---

## 8. Definición matemática del desempeño

En `performance.py`, definiciones D1–D9 listas para copiar a la tesis.

**Marco común.** Ideal/nadir verdaderos derivados de Z, idénticos para todos los
métodos y corridas. **Nunca el z_ref adaptativo del algoritmo** — medir HV con el
z_ref propio hace el desempeño trivialmente maximizable inflando z_ref, que es
exactamente el modo de falla que la cota de la sección 5 previene.

- **D1 — HVR(A) = HV(Â,z^ext)/HV(Ẑ,z^ext) ∈ [0,1]**, con z^ext=(1+κ)·1, κ=0.1.
  Comparable **entre problemas** y con significado absoluto: "fracción del
  hipervolumen alcanzable capturada".
- **D2 — IGD+** contra Ẑ en el marco normalizado.
- **D3 — Eratio(A) = E^ln_s(Â)/E^ln_s(U_{|Â|})** contra un subconjunto uniforme del
  mismo tamaño sobre el mismo frente. Elimina la dependencia de m, |A| y escala.
- **D4 — Anytime: AT(A) = (1/T)Σ_t HVR(A(t))**, área bajo la curva. Es lo que un
  *calendario* de punto de referencia debería mejorar: dos métodos pueden terminar
  en el mismo HVR y diferir mucho en qué tan rápido llegaron.
- **D5 — Tiempo al objetivo** T_q.
- **D6–D8 —** mediana + IQR (no media: las distribuciones sobre semillas son
  sesgadas), rangos por bloque, Friedman y Quade.
- **D9 —** "a supera a b" ⟺ Wilcoxon con Bonferroni rechaza **y** el signo favorece
  a *a*, reportando **tamaño de efecto** (correlación rango-biserial). Con R=30, una
  diferencia en la cuarta decimal puede ser significativa e irrelevante a la vez.

---

## 9. Pruebas con los 8 problemas IMOP

`imop.py` porta **IMOP1–IMOP8 fielmente desde el MATLAB de PlatEMO**, tanto
`CalObj` como `GetOptimum`. Validados contra muestreo denso no dominado de la
variedad g=0 (distancia media 0.0001–0.016).

> **Importante para el Cuadro 1.** IMOP1–3 son **biobjetivo** e IMOP4–8
> triobjetivo por definición del benchmark. La suite **no está parametrizada en m**.
> El Cuadro 1 pedía un IMOP3 de 3 objetivos, que no existe en la literatura; el
> código anterior lo suplía con `IMOP3Like`, un sustituto inventado.
> `problem_n_obj()` ahora devuelve el m correcto y lanza error explícito si se pide
> otro, en vez de fabricar una variante inexistente en silencio.

Se necesitó un filtro no dominado **por bloques** (exacto, memoria acotada) porque
el de pymoo es O(n²) y las retículas de 10⁵ puntos que hacen falta para resolver
los frentes disconexos de IMOP3/6/8 lo tumbaban.

---

## 10. Baselines: reemplazo de `SMS-EMOA_nadir`

**Corrección de atribución.** R2-EMOA y el SMS-EMOA adaptativo al nadir son dos
algoritmos distintos de dos grupos distintos:

- **SMS-EMOA (y su punto de referencia adaptativo)** — N. Beume, B. Naujoks,
  M. Emmerich, *SMS-EMOA: Multiobjective selection based on dominated hypervolume*,
  EJOR 181(3):1653–1669, 2007.
- **R2-EMOA** — H. Trautmann, T. Wagner, D. Brockhoff, *R2-EMOA: Focused
  Multiobjective Search Using R2-Indicator-Based Selection*, LION 7, LNCS 7997,
  pp. 70–74, 2013 (versión extendida: D. Brockhoff, T. Wagner, H. Trautmann,
  *R2 Indicator Based Multiobjective Search*, Evol. Comput. 23(3):369–395, 2015).

**Por qué se reemplazó el baseline anterior.** `SMS-EMOA_nadir` usaba
z_ref=(1.01,…,1.01) en el marco normalizado: un *hombre de paja*. Pone el punto tan
cerca del nadir que las soluciones extremas casi no contribuyen, y no es lo que
Beume et al. proponen. En DTLZ2 obtenía HVR≈0.28 contra ≈0.88 del resto, así que
"RL-RP supera a nadir" era una afirmación débil.

Ahora hay dos baselines honestos (`r2_emoa.py`):

- **`SMS-EMOA_nadir-adaptive`** — z_ref = peor valor objetivo de la población
  actual + offset, en el espacio **crudo**. En el marco normalizado equivale a
  `z_norm = 1 + offset/(nadir−ideal)`, es decir se auto-escala con la extensión del
  frente. Es el baseline de hipervolumen que hay que vencer.
- **`R2-EMOA`** — control agnóstico al indicador. **No tiene punto de referencia
  que adaptar**, y ésa es precisamente su utilidad: responde una pregunta que
  ninguno de los otros puede — cuánto de la diferencia se debe al punto de
  referencia *en absoluto*, frente a que la selección por hipervolumen sea la
  herramienta equivocada en frentes irregulares. Si R2-EMOA gana en
  Minus-DTLZ/IMOP, ajustar z_ref es optimizar la perilla equivocada, y eso es un
  hallazgo más valioso que una ganancia marginal en HV.

Efecto inmediato del cambio (DTLZ2, T=1500, μ=30, 2 semillas):

```
metodo                       HVR     IGD+   Eratio    s
SMS-EMOA_nadir-adaptive   0.8778   0.0408    1.538  13.8
SMS-EMOA_balanced         0.8815   0.0389    1.301  14.2
SMS-EMOA_dynlin           0.8770   0.0412    1.758  13.8
R2-EMOA                   0.8197   0.0534    1.732  11.7
RL-RP-SMS-EMOA            0.8445   0.0656    2.823  21.5
```

**Con el baseline honesto, RL-RP ya no lo supera.** El baseline antiguo hacía
parecer que sí. `LEGACY_METHODS` conserva el viejo para reproducir resultados
previos.

---

## 11. Guardar todos los valores de todas las generaciones

`history_io.py`: un archivo por (problema, método, semilla) en
`<histdir>/<problema>/<método>/seedNNN.csv.gz`, con ~40 columnas por generación —
z_ref completo, ideal/nadir, HV adaptativo y fijo, dispersión, energía, γ, HVR,
IGD+, Eratio, régimen, subacción, índice de estado, recompensa, ε, σ₁₋₃, pagos y
las ocho variables de estado. Metadatos del run (|S|, t_adapt, cobertura Q) en un
`.meta.json` adjunto.

Dos parámetros distintos porque el costo no está donde parece:

- `record_every` — adelgaza el registro (por defecto **1**: todas las generaciones).
- `eval_every` — adelgaza **sólo** los indicadores caros. Calcular IGD+ contra 5000
  puntos 100 000 veces por corrida cuesta mucho más que la búsqueda misma.

Costo en disco: ~3–4 MB por corrida, ~3–4 GB para la rejilla completa. El tiempo,
no el disco, es la restricción.

---

## 12. Complejidad muestral de la tabla Q — el hallazgo principal

### Diagnóstico (DTLZ2, μ=40)

```
                              T_max=2000        T_max=10000
estados distintos visitados      19 (2.8%)        43 (6.4%)
número EFECTIVO de estados        5.4              7.8
entradas Q con ≥1 update         50 (1.2%)        13 (0.32%)
updates totales                 747              254
DECISIONES CON FILA Q NULA     47.7%            96.5%
uso de regímenes           bal=53% coarse=38%   coarse=95%
σ final                      (0, 1, 0)          (0, 0, 1)
z_ref final              (3.9, 1.1, 4.6)    (7.0, 9.0, 9.0)
```

Tres defectos acoplados:

- **(a) |S|=675 es absurdo.** Sólo se visitan 19–43 estados y el número *efectivo*
  (exponencial de la entropía) es **5–8**.
- **(b) La Ec. 10 mata dos tercios de la tabla.** Al actualizar Q sólo en el
  régimen *balanced*, `Q[:,fine,:]` y `Q[:,coarse,:]` quedan idénticamente cero
  para siempre; cuando σ favorece esos regímenes la dirección se elige **al azar**.
- **(c) σ colapsa a un vértice y no regresa.** La actualización replicadora es
  pesos multiplicativos sobre el símplex y los vértices son absorbentes.

Se retroalimentan: σ colapsa a *coarse* → Q nunca se actualiza → 96.5% de
decisiones ciegas → caminata aleatoria con δ=1 → z_ref satura la caja. **Con más
presupuesto hay *menos* aprendizaje** (747 → 254 updates).

### Correcciones implementadas

- `q_update_regimes="all"` (default) — cada régimen aprende de su propia experiencia.
- `sigma_floor=0.05` — ecuación replicador-**mutador**, σ ← (1−η)σ + η/K. Acota
  σ_k ≥ η/K y mantiene vivas todas las estimaciones de pago.
- `action_every=τ` — el planeador actúa cada τ generaciones y la recompensa se mide
  sobre toda la ventana.
- Diagnóstico `frac_blind` para que esto sea medible y no supuesto.

### Resultado contraintuitivo

```
configuración                        ciegas%  updates  cob.%     HVR    IGD+
original (Ec.10 estricta, sin piso)     88.7      319   0.22  0.7580  0.1024
+ Q en todos los regímenes               1.3     2799   0.86  0.5538  0.2318
+ piso mutador en sigma                 73.2      767   1.33  0.8845  0.0382
+ ambos                                  2.5     2799   1.42  0.4300  0.2931
+ ambos + estado grueso (243)            2.0     2799   3.38  0.5945  0.2119
```

**Hacer que Q realmente aprenda EMPEORA el desempeño** (0.758 → 0.554 → 0.430).
Lo único que ayuda claramente es el piso de σ *sin* arreglar el aprendizaje.

### Causa raíz: la recompensa no tiene señal

Correlación de Spearman entre r_t y la mejora real posterior en HVR:

```
Q aprende:  ρ = +0.008
original:   ρ = −0.028
```

**La recompensa no contiene información sobre lo que importa.** Aprenderla mejor
perjudica porque se optimiza ruido.

*Motivo:* desajuste de escalas temporales. En un SMS-EMOA steady-state una
generación crea **un** hijo y elimina **un** individuo: el ΔHV de una generación
está dominado por el ruido de los operadores de variación. El efecto de mover z_ref
es indirecto y se acumula a lo largo de un recambio poblacional completo (~μ
generaciones). Con γ=0.9 el horizonte efectivo es 1/(1−γ)=10 generaciones; el
efecto tarda ~100. **El horizonte de asignación de crédito está mal por uno o dos
órdenes de magnitud.**

### Experimento de τ (DTLZ2, T=4000, μ=30, 3 semillas)

```
configuración                         ciegas%    upd  cob.%     HVR    IGD+   z_ref
tau=1   |S|=675 (por generación)          2.5   2799   1.42  0.4310  0.2936   3.608
tau=30  |S|=675                          38.7     92   0.44  0.5682  0.2293   4.153
tau=30  |S|=32  (2x2x2x2x2)              15.1     92   6.08  0.5701  0.2288   3.440
tau=100 |S|=32                           46.4     27   3.30  0.3997  0.3577   1.973
tau=30  |S|=32  SÓLO piso sigma          68.8     31   4.69  0.5715  0.2285   3.004
(ref) SMS-EMOA_balanced                     -      -      -  0.8955  0.0353
(ref) SMS-EMOA_dynlin                       -      -      -  0.8956  0.0355
```

Lecturas:

1. **τ=30 mejora claramente sobre τ=1** (0.431 → 0.568/0.570). La hipótesis de la
   escala temporal se sostiene.
2. **τ=100 empeora** (0.400): sólo 27 updates, se pasa al otro extremo.
3. **Reducir |S| de 675 a 32 no cambia nada a τ=30** (0.5682 vs 0.5701), lo que
   confirma que |S| nunca fue el cuello de botella.
4. **Ningún variante de RL-RP alcanza a los baselines fijos** (≈0.895) en esta
   configuración. La mejor variante encontrada hasta ahora es *sólo piso de σ, τ=1,
   |S|=675* con HVR≈0.885 — es decir, **empata con un punto de referencia fijo
   trivial**.

> **Advertencia estadística.** Son 3 semillas. Diferencias menores a ~0.1 en HVR no
> son confiables. Las corridas sí son reproducibles (configuraciones repetidas dan
> 0.4310 vs 0.4300), pero el ranking necesita el protocolo completo con 30 semillas
> y las pruebas de la Definición D9.

### Conclusión para la tesis

El problema no es la cantidad de aprendizaje sino la **especificación de la
recompensa**. Es una crítica al diseño del método, no al código, y explica por qué
RL-RP resulta competitivo pero no superior. No se ve mirando sólo indicadores
finales — hizo falta guardar todas las generaciones para detectarlo.

**Siguiente paso sugerido:** barrer τ ∈ {1,10,30,50,100} midiendo simultáneamente
la correlación recompensa–ΔHVR y el HVR final. Si la correlación sube con τ y el
HVR la sigue, tienes diagnóstico y solución en la misma figura.

---

## 13. Otros bugs corregidos

- `hypervolume` calculaba la máscara de puntos dominantes y **la ignoraba**.
- `select_subaction` usaba `argmax` sobre una fila de ceros → devolvía siempre la
  acción 0. Ahora desempata al azar.
- `sms_emoa_eliminate`: cuando ningún miembro del peor frente domina a z_ref, todas
  las contribuciones son 0 y `argmin` devolvía determinísticamente el índice 0 — la
  selección degeneraba en "borrar siempre al primero". Ahora hay criterio secundario.
- `tests_reference_sets.py`: pymoo espera los parámetros posicionales de WFG en
  rango unitario y aplica el escalado por `xu` internamente.
