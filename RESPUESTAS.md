# Respuestas a las observaciones del seminario — RL-RP-SMS-EMOA

Documento de acompañamiento al código. Cada sección responde una observación y
apunta al archivo donde quedó implementada. Las secciones 1–10 siguen el orden
en que se hicieron los comentarios; las 11–14 y 16 son los cambios
adicionales, y la **sección 15 reporta los resultados del protocolo completo**
(1 680 corridas en el clúster ixachi).

---

## 1. ¿Cómo se manejará el valor de diversidad?

**Problema encontrado.** Dos de las cinco variables de estado se normalizaban
con un **máximo corriente**: `D_norm(t)=D(t)/max_{t'≤t}D(t')`,
`HV_norm(t)=HV(t)/max_{t'≤t}HV(t')`. Un máximo corriente es un estadístico de
la *trayectoria*, no de la población: el mismo frente podía ser el estado 137 en
t=1000 y el 42 en t=50000. Eso vuelve el proceso de decisión **no
estacionario**, y Q-learning tabular sólo tiene garantías sobre un MDP
estacionario.

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

## 2. Dinámica replicadora: 0.01, 1/H, 1 para mover el punto. ¿Está normalizado el espacio objetivo?

Sí, pero **el nadir estaba mal estimado**: `estimate_ideal_nadir` tomaba el
máximo sobre **toda la población**, no sobre el frente no dominado, de modo que
los peores individuos dominados dictaban la escala. Corregido a máximo
componente a componente sobre el frente no dominado
(`indicators.estimate_ideal_nadir`, `from_front=True`).

Con eso los pasos (0.01, 1/H, 1.0) sí son desplazamientos en unidades
normalizadas: 1 % de la extensión del frente, el espaciado clásico de Ishibuchi
con H=12, y un rango nadir–ideal completo.

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
R2-EMOA se mantiene como control: su selección cuesta O(|W|·|A|·m) y no explota
con m (sección 12).

**Pero el límite real no es m, es la complejidad muestral** — ver sección 14.

---

## 4. Retractarse de una decisión mala — signo para mover

Implementado y es la razón de ser del esquema de acciones
(`rl_planner.decode_action`, `rl_planner.apply_action`). Una acción es un par
(dimensión, signo) `(i, s) ∈ {1..m} × {−1,+1}`, codificado como
`j = 2(i−1) + 𝟙[s=+1]`, y sólo modifica la componente i:

```
z_i(t+1) = clip(z_i(t) + s·δ_k, 1+ε, z_max),    z_{i'}(t+1) = z_{i'}(t)
```

Con `s = −1` el planeador revierte, total o parcialmente, un desplazamiento
previo. En el esquema original `z_ref = ẑ_n + δ·1` esto es imposible: δ sólo
puede crecer y en todas las dimensiones a la vez.

**Consecuencia medible.** Que el movimiento sea reversible es lo que permite
que `z_ref(t)` sea una trayectoria y no una rampa monótona; se registra completa
(columnas `zref1..zrefm` de cada generación) precisamente para poder contrastar
si el signo elegido corrige o amplifica el error anterior.

---

## 5. Dos fases: adaptación 70 %, refinación 30 %

Implementado con `rho = 0.7` (`core.DEFAULTS`, `algorithm.RLRPSMSEMOA`):

- `t_adapt = ⌈ρ·T_max⌉`. Para `t ≤ t_adapt` se ejecuta el ciclo completo
  (estado → dinámica replicadora → Q-learning → movimiento de `z_ref`).
- Para `t > t_adapt`, `z_ref` queda congelado en `z_ref(t_adapt)`; no hay
  cálculo de estado, ni muestreo de acciones, ni actualizaciones de Q. SMS-EMOA
  opera como el algoritmo estándar con PR constante.

Cada fila de la historia lleva la bandera `adapting`, así que el corte entre
fases es visible en los datos y no hay que inferirlo. `rho` es barrible con
`run_sensitivity.py --param rho --values 0.5 0.7 0.9` (Sec. 5.3 c).

---

## 6. ¿Cómo se detecta la geometría en el estado? No es muy claro

**Antes no se detectaba.** `ĝ` (Ec. 6) mide la razón de contribuciones al HV
entre la mitad exterior e interior del frente: eso es *dónde está la presión de
selección*, no *qué forma tiene el frente*. Peor, **depende de z_ref**, que es
justo lo que el planeador mueve, así que confundía causa con efecto.

**Corrección** (`indicators.estimate_curvature_p`, `geometry_gamma`). Se ajusta
el exponente de curvatura *p* del modelo `Σ_i f̂_i^p = 1` por bisección
geométrica sobre la mediana, y se usa

> **γ = p/(1+p) ∈ (0,1)** — γ<½ convexo, γ=½ lineal, γ>½ cóncavo

acotado por construcción, sin normalización corriente. Valores medidos sobre los
frentes analíticos reales (|Z| ≈ 2000), valor de *p*:

| DTLZ1 | DTLZ2 | −DTLZ1 | −DTLZ2 | WFG4 | WFG9 |
|---|---|---|---|---|---|
| 1.00 | 2.00 | 3.88 | 2.43 | 2.01 | 2.00 |

| IMOP1 | IMOP2 | IMOP3 | IMOP4 | IMOP5 | IMOP6 | IMOP7 | IMOP8 |
|---|---|---|---|---|---|---|---|
| 0.25 | 4.00 | 0.83 | 1.59 | 1.86 | 1.75 | 2.00 | 1.58 |

El descriptor separa correctamente los tres regímenes: IMOP1 sale fuertemente
convexo (p=0.25), IMOP2 fuertemente cóncavo (p=4.00), DTLZ1 exactamente lineal.

Se añadió además **ι (invertidez)** = percentil bajo de `Σ_i f̂_i` menos 1,
normalizado: distingue frentes regulares (ι≈0) de invertidos (Minus-DTLZ1
ι≈0.50). Se registra como diagnóstico; incorporarlo al estado multiplicaría |S|,
lo cual es contraproducente dada la sección 14.

La formulación original (Ec. 6) sigue disponible como
`geometry_mode="contour"` (y `"both"`) para el análisis de ablación que la
Sec. 6.1 de la propuesta compromete.

---

## 7. ¿Hasta dónde se puede llegar en el movimiento? ¿Está acotado?

**No lo estaba por arriba.** La actualización era `z_i ← max(1, z_i + s·δ)`. Con
el régimen grueso (δ=1) el punto podía irse a 10³ o más, donde el paisaje de HV
es plano y SMS-EMOA degenera en un método que empuja la población a los
extremos.

**Corrección** (`rl_planner.apply_action`), siguiendo exactamente la sugerencia
recibida de acotar con un punto adicional tipo (10,10,…,10):

```
z_i(t+1) = clip( z_i(t) + s·δ_k , 1+ε , z_max )    ε = 1e-3,  z_max = 10
```

En el espacio objetivo normalizado el nadir es (1,…,1), de modo que la caja
`[1+ε, 10]^m` es precisamente "entre el nadir y el punto (10,…,10)". La cota
inferior es estrictamente **mayor** que 1 para que las soluciones extremas
conserven contribución positiva al HV (con z_i=1 exacto, una solución que
alcanza el nadir en el objetivo i contribuye volumen cero y sería la primera
eliminada).

*Alcanzabilidad:* desde z=1 el régimen grueso llega a z_max en 9 pasos, el
balanceado en 108, el fino en 900 — despreciable frente a ρ·T_max ≈ 7×10⁴. La
caja restringe **dónde puede terminar**, no **qué puede explorar**.

*Evidencia empírica:* sin la cota, z_ref satura. En corridas medidas terminó en
(7.0, 9.0, 9.0) y (3.9, 1.1, 4.6) — pegado a la frontera. Causa: con Q sin
entrenar y σ₀=(0.10,0.10,0.80), el punto hace una **caminata aleatoria** con
pasos ±1.

`z_max` es configurable (`--zref_max`) y barrible, por si conviene estudiar el
efecto de la cota misma.

---

## 8. IGD+ — ¿qué conjunto de referencia?

Se encontraron **dos bugs graves**.

**(a) Fallback de muestreo aleatorio.** Para WFG e IMOP, Z se generaba
muestreando el espacio de decisión al azar y filtrando no dominados. Eso **no es
una muestra del frente de Pareto**: muestreo uniforme de un WFG de 14 variables
prácticamente nunca alcanza g=0. Los valores "IGD+" reportados no eran IGD+.

**(b) El frente de Minus-DTLZ estaba mal.** El código hacía `PF_minus = −PF_base`.
Como `f = (1+g)·h(x_pos)`, minimizar −f exige g en su **máximo**, no en 0:

> **PF(Minus-DTLZ) = −(1+g_max) · PF(DTLZ)**

Factores verificados numéricamente: **Minus-DTLZ2 → 3.5**,
**Minus-DTLZ1 → 1102.30**. El conjunto de referencia estaba equivocado por un
factor de 3.5 y por *tres órdenes de magnitud* respectivamente; por eso IGD+
salía exactamente 0.0 (la aproximación dominaba todo Z).

**Ahora Z es siempre analítico** (`problems.reference_set`): pymoo con retícula
Das-Dennis para DTLZ/WFG, negación escalada para Minus-DTLZ, y el `GetOptimum`
original de PlatEMO para IMOP. Validación completa
(`python3 -m rlrp_smsemoa.tests_reference_sets`, corrida sobre los 14 problemas):

```
problema        m    |Z|  Z dominados  d_rel medio  d_rel p95
dtlz1           3   1953            0      0.00471    0.00693
dtlz2           3   1953            0      0.00632    0.01048
minus-dtlz1     3   1953            0      0.00471    0.00693
minus-dtlz2     3   1953            0      0.00632    0.01048
wfg4            3   1953            0      0.00536    0.00988
wfg9            3   1953            0      0.00602    0.01088
imop1           2   2000            0      0.00017    0.00032
imop2           2   2000            0      0.00015    0.00028
imop3           2   2000            0      0.00005    0.00016
imop4           3   2000            0      0.00020    0.00045
imop5           3   2024            0      0.00069    0.00347
imop6           3   2000            0      0.00223    0.00399
imop7           3    732            0      0.00288    0.00531
imop8           3   2000            0      0.00657    0.02636
```

Cero puntos dominados en todos los casos y distancia a la variedad óptima real
por debajo del 1 % del diámetro (salvo IMOP8, 2.6 %, el multimodal extremo).

> **|Z| debe reportarse siempre**: IGD+ es un estadístico muestral de Z y
> valores con distinto |Z| no son comparables. Queda guardado como `n_ref` en el
> `.meta.json` de cada corrida.

---

## 9. ¿Cómo influye el valor de la energía de Riesz en la decisión del movimiento?

Entra por **tres vías**, y había un bug en la primera:

1. **Estado.** `E_norm` = razón de energías logarítmicas consecutivas. El código
   hacía `min(ratio,1.0)` (Ec. 5 literal) — **el recorte hacía invisible el
   deterioro de uniformidad**; el planeador nunca podía ver que un movimiento
   había empeorado la distribución. Ahora `E_norm = clip(ratio,0,2)/2`, donde
   ½ = sin cambio, <½ mejoró, >½ empeoró.
2. **Recompensa.** `r_t = ΔHV + α·(E_norm(t) − E_norm(t+1))`, con α=1 por
   defecto (`alpha_reward`).
3. **Pagos de la dinámica replicadora.**
   `u_k = β(D_norm)·EMA[ΔU]_k + (1−β)·EMA[ΔHV]_k`, con
   `β_t = 0.9 − 0.8·D_norm(t)`: cuando la diversidad es baja el pago pesa la
   uniformidad (exploración) y cuando es alta pesa el HV (explotación). En el
   código original esta variable se llamaba `delta_D` (dispersión) pero contenía
   una diferencia de **energía**; renombrada a `delta_U` y documentada.

La energía pasó a ser **promedio por par** (dividida entre n(n−1)); la suma
cruda escala como n², así que un cambio en |A| movía la variable de estado
aunque la geometría no cambiara.

**Cuestión de diseño abierta.** La recompensa usaba HV contra el z_ref *móvil*:
como `HV(A,z_ref)/vol(z_ref)` cambia al mover z_ref aunque la población esté
intacta, el planeador se premiaba en parte por **mover la portería**. Se
implementó `reward_hv="fixed"` (por defecto) y `"adaptive"` (el original) para
compararlos.

---

## 10. Falta definir matemáticamente el "desempeño" de los algoritmos

En `performance.py`, definiciones D1–D9 listas para copiar a la tesis.

**Marco común.** Ideal/nadir verdaderos derivados de Z, idénticos para todos los
métodos y corridas. **Nunca el z_ref adaptativo del algoritmo** — medir HV con
el z_ref propio hace el desempeño trivialmente maximizable inflando z_ref, que
es exactamente el modo de falla que la cota de la sección 7 previene.

- **D1 — HVR(A) = HV(Â,z^ext)/HV(Ẑ,z^ext) ∈ [0,1]**, con z^ext=(1+κ)·1, κ=0.1.
  Comparable **entre problemas** y con significado absoluto: "fracción del
  hipervolumen alcanzable capturada".
- **D2 — IGD+** contra Ẑ en el marco normalizado.
- **D3 — Eratio(A) = E^ln_s(Â)/E^ln_s(U_{|Â|})** contra un subconjunto uniforme
  del mismo tamaño sobre el mismo frente. Elimina la dependencia de m, |A| y
  escala.
- **D4 — Anytime: AT(A) = (1/T)Σ_t HVR(A(t))**, área bajo la curva. Es lo que un
  *calendario* de punto de referencia debería mejorar: dos métodos pueden
  terminar en el mismo HVR y diferir mucho en qué tan rápido llegaron.
- **D5 — Tiempo al objetivo** T_q, q=0.95 del mejor HVR alcanzado por cualquier
  método en esa (problema, semilla).
- **D6–D8 —** mediana + IQR (no media: las distribuciones sobre semillas son
  sesgadas), rangos por bloque, Friedman y Quade.
- **D9 —** "a supera a b" ⟺ Wilcoxon con Bonferroni rechaza **y** el signo
  favorece a *a*, reportando **tamaño de efecto** (correlación rango-biserial).
  Con R=30, una diferencia en la cuarta decimal puede ser significativa e
  irrelevante a la vez.

---

## 11. Pruebas con los ocho problemas IMOP

`imop.py` porta **IMOP1–IMOP8 fielmente desde el MATLAB de PlatEMO**, tanto
`CalObj` como `GetOptimum`, y **los ocho entran en el conjunto de prueba**
(`problems.BENCHMARK` = 6 regulares + 8 IMOP = 14 problemas;
`run_full_experiment.py --problems all`, o `--problems imop` para reportarlos
por separado como pide la Sec. 6.1).

> **Importante para el Cuadro 1.** IMOP1–3 son **biobjetivo** e IMOP4–8
> triobjetivo por definición del benchmark. La suite **no está parametrizada en
> m**. El Cuadro 1 pedía un IMOP3 de 3 objetivos, que no existe en la
> literatura; el código anterior lo suplía con `IMOP3Like`, un sustituto
> inventado. `problem_n_obj()` ahora devuelve el m correcto y lanza error
> explícito si se pide otro, en vez de fabricar una variante inexistente en
> silencio.

Se necesitó un filtro no dominado **por bloques** (exacto, memoria acotada)
porque el de pymoo es O(n²) y las retículas de 10⁵ puntos que hacen falta para
resolver los frentes disconexos de IMOP3/6/8 lo tumbaban.

También se corrigió la documentación de IMOP1/IMOP2, que tenía las geometrías
intercambiadas: IMOP1 (cos⁸/sen⁸) es **convexo** (p=0.25) e IMOP2 (cos^½/sen^½)
es **cóncavo** (p=4.00).

---

## 12. Reemplazo de `SMS-EMOA_nadir` por **d-SMS-EMOA**

`SMS-EMOA_nadir` usaba z_ref = (1.01,…,1.01) en el marco normalizado: un
*hombre de paja*. Pone el punto tan cerca del nadir que las soluciones extremas
casi no contribuyen, y no es lo que propone ninguno de los trabajos citados. En
DTLZ2 obtenía HVR≈0.28 contra ≈0.88 del resto, así que "RL-RP supera a nadir"
era una afirmación vacía.

En su lugar se implementó el esquema publicado de especificación **dinámica**
del punto de referencia (`algorithm.run_d_sms_emoa`):

> **d-SMS-EMOA** (también SMS-EMOA-DRP) — H. Ishibuchi, R. Imada, N. Masuyama,
> Y. Nojima, *Dynamic specification of a reference point for hypervolume
> calculation in SMS-EMOA*, CEC 2018, pp. 701–708.

El punto z_ref = (r,…,r) en el espacio objetivo **normalizado** se desplaza de
**r = 10** en la población inicial a **r = 1** en la final:

```
r(t) = 10 + (t−1)/(T_max−1) · (1 − 10)
```

Un r grande hace que las soluciones extremas concentren el hipervolumen, de modo
que la búsqueda primero se abre hacia la **frontera** del frente; un r pequeño
hace que dominen las interiores y la búsqueda se concentra en el **centro**.
Recorrer r de 10 a 1 visita ambos regímenes en el orden que importa para un
frente no triangular, que es justamente el caso donde la especificación del PR
—y no el algoritmo— decide la forma del conjunto final.

Dos precisiones honestas sobre la implementación:

- El artículo especifica los extremos (r=10 inicial, r=1 final). Interpolar
  **linealmente en el índice de generación** es la lectura de este código;
  `r_start` y `r_end` quedan expuestos como parámetros por si conviene otra
  interpolación.
- r=1 pone el PR exactamente sobre el nadir estimado, donde una solución que
  alcanza el nadir contribuye volumen cero. Es la especificación del artículo y
  se respeta; `sms_emoa.sms_emoa_eliminate` tiene un criterio secundario
  explícito para ese caso degenerado.

Esto reemplaza también al antiguo `SMS-EMOA_dynlin`, que pretendía implementar
el mismo artículo pero movía el PR de 1.01 a 2.0, es decir **en la dirección
contraria** y con la magnitud equivocada.

### Las cuatro configuraciones comparadas

| Método | Punto de referencia | Fuente |
|---|---|---|
| `SMS-EMOA_balanced` | fijo, (1+1/H)·1 | Ishibuchi et al., GECCO 2017 |
| `d-SMS-EMOA` | dinámico, r de 10 a 1 | Ishibuchi et al., CEC 2018 |
| `R2-EMOA` | ninguno (selección R2) | Trautmann, Wagner & Brockhoff, LION 2013 |
| `RL-RP-SMS-EMOA` | aprendido en línea | propuesto |

**R2-EMOA** se conserva como control agnóstico al indicador. **No tiene punto de
referencia que adaptar**, y ésa es precisamente su utilidad: responde una
pregunta que ninguno de los otros puede — cuánto de la diferencia se debe al
punto de referencia *en absoluto*, frente a que la selección por hipervolumen
sea la herramienta equivocada en frentes irregulares. Si R2-EMOA gana en
Minus-DTLZ/IMOP, ajustar z_ref es optimizar la perilla equivocada, y eso es un
hallazgo más valioso que una ganancia marginal en HV.

**Advertencia sobre el resultado.** Con baselines honestos, RL-RP **no** los
supera. Con el baseline de paja parecía que sí. Es un resultado, no un fallo del
código; la sección 14 explica por qué.

---

## 13. Guardar todos los valores, no sólo los finales

Implementado sin condiciones: `core.record` se llama **en cada generación** de
**cada método**, sin adelgazamiento, y `experiment.run_single` **siempre**
escribe la historia a disco. No hay modo "sólo el final".

`history_io.py`: un archivo por (problema, método, semilla) en
`<histdir>/<problema>/<método>/seedNNN.csv.gz`, con ~40 columnas por generación
— z_ref completo, ideal/nadir, HV adaptativo y fijo, dispersión, energía, γ,
HVR, IGD+, Eratio, régimen, subacción, índice de estado, recompensa, ε, σ₁₋₃,
pagos y las ocho variables de estado. Metadatos del run (|S|, |Z|, t_adapt,
cobertura Q) en un `.meta.json` adjunto.

Queda un solo parámetro, y no toca el registro:

- `eval_every` — adelgaza **sólo** los indicadores externos caros (HVR/IGD+/
  Eratio contra Z). Por defecto **1**: tampoco se adelgaza nada. Subirlo sólo
  tiene sentido en el barrido de µ grande de la Sec. 5.3(b).

(El antiguo `record_every` se eliminó: permitía justamente lo que la observación
pedía evitar.)

Costo en disco: ~3–4 MB por corrida, ~5–7 GB para la rejilla completa de
14 problemas × 4 métodos × 30 semillas. El tiempo, no el disco, es la
restricción.

**No es contabilidad ociosa:** el hallazgo de la sección 14 es invisible en una
tabla de valores finales y sólo aparece en las trayectorias.

---

## 14. Complejidad muestral de la tabla Q — el diagnóstico

> Los experimentos de esta sección son diagnósticos cortos (2–3 semillas) que
> explican *por qué* falla el mecanismo. La confirmación con el protocolo
> completo (30 semillas) está en la sección 15, y coincide.

### Diagnóstico (DTLZ2, µ=40)

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

- **(a) |S|=675 es absurdo.** Sólo se visitan 19–43 estados y el número
  *efectivo* (exponencial de la entropía) es **5–8**.
- **(b) La Ec. 10 mata dos tercios de la tabla.** Al actualizar Q sólo en el
  régimen *balanced*, `Q[:,fine,:]` y `Q[:,coarse,:]` quedan idénticamente cero
  para siempre; cuando σ favorece esos regímenes la dirección se elige **al
  azar**.
- **(c) σ colapsa a un vértice y no regresa.** La actualización replicadora es
  pesos multiplicativos sobre el símplex y los vértices son absorbentes.

Se retroalimentan: σ colapsa a *coarse* → Q nunca se actualiza → 96.5 % de
decisiones ciegas → caminata aleatoria con δ=1 → z_ref satura la caja. **Con más
presupuesto hay *menos* aprendizaje** (747 → 254 updates).

### Correcciones implementadas

- `q_update_regimes="all"` (default) — cada régimen aprende de su propia
  experiencia. `"balanced"` reproduce la Ec. 10 literal.
- `sigma_floor=0.05` — ecuación replicador-**mutador**, σ ← (1−η)σ + η/K. Acota
  σ_k ≥ η/K y mantiene vivas todas las estimaciones de pago. `0` recupera la
  Ec. 9 literal.
- `action_every=τ` — el planeador actúa cada τ generaciones y la recompensa se
  mide sobre toda la ventana.
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
está dominado por el ruido de los operadores de variación. El efecto de mover
z_ref es indirecto y se acumula a lo largo de un recambio poblacional completo
(~µ generaciones). Con γ=0.9 el horizonte efectivo es 1/(1−γ)=10 generaciones;
el efecto tarda ~100. **El horizonte de asignación de crédito está mal por uno o
dos órdenes de magnitud.**

### Experimento de τ (DTLZ2, T=4000, µ=30, 3 semillas)

```
configuración                         ciegas%    upd  cob.%     HVR    IGD+   z_ref
tau=1   |S|=675 (por generación)          2.5   2799   1.42  0.4310  0.2936   3.608
tau=30  |S|=675                          38.7     92   0.44  0.5682  0.2293   4.153
tau=30  |S|=32  (2x2x2x2x2)              15.1     92   6.08  0.5701  0.2288   3.440
tau=100 |S|=32                           46.4     27   3.30  0.3997  0.3577   1.973
tau=30  |S|=32  SÓLO piso sigma          68.8     31   4.69  0.5715  0.2285   3.004
(ref) SMS-EMOA_balanced                     -      -      -  0.8955  0.0353
```

Lecturas:

1. **τ=30 mejora claramente sobre τ=1** (0.431 → 0.568/0.570). La hipótesis de
   la escala temporal se sostiene.
2. **τ=100 empeora** (0.400): sólo 27 updates, se pasa al otro extremo.
3. **Reducir |S| de 675 a 32 no cambia nada a τ=30** (0.5682 vs 0.5701), lo que
   confirma que |S| nunca fue el cuello de botella.
4. **Ninguna variante de RL-RP alcanza al baseline fijo** (≈0.895) en esta
   configuración. La mejor encontrada hasta ahora es *sólo piso de σ, τ=1,
   |S|=675* con HVR≈0.885 — es decir, **empata con un punto de referencia fijo
   trivial**.

> **Advertencia estadística.** Son 3 semillas. Diferencias menores a ~0.1 en HVR
> no son confiables. Las corridas sí son reproducibles (configuraciones
> repetidas dan 0.4310 vs 0.4300), pero el ranking necesita el protocolo
> completo con 30 semillas y las pruebas de la Definición D9.

### Conclusión para la tesis

El problema no es la cantidad de aprendizaje sino la **especificación de la
recompensa**. Es una crítica al diseño del método, no al código, y explica por
qué RL-RP resulta competitivo pero no superior. No se ve mirando sólo
indicadores finales — hizo falta guardar todas las generaciones para detectarlo.

**Siguiente paso sugerido:** barrer τ ∈ {1,10,30,50,100} midiendo
simultáneamente la correlación recompensa–ΔHVR y el HVR final
(`run_sensitivity.py --param action_every`). Si la correlación sube con τ y el
HVR la sigue, hay diagnóstico y solución en la misma figura.

---

## 15. Resultados del protocolo completo (14 problemas × 4 métodos × 30 semillas)

Corrida en el clúster ixachi: 1 680 ejecuciones, T_max = 100 000, µ = 100,
todas las generaciones guardadas. Medianas sobre 30 semillas.

### Cuántas veces gana cada método (mejor mediana, 14 problemas)

| Indicador | balanced | R2-EMOA | d-SMS-EMOA | RL-RP |
|---|---|---|---|---|
| HVR (D1) | **6** | 5 | 2 | 1 |
| IGD+ (D2) | **6** | 5 | 2 | 1 |
| Eratio (D3) | **11** | 0 | 0 | 3 |
| Anytime (D4) | **5** | 4 | 4 | 1 |

**El punto de referencia fijo trivial (1+1/H) es el método más fuerte.** Con
los baselines honestos y el protocolo completo, RL-RP-SMS-EMOA no supera a
ninguno de ellos de forma sistemática.

### El hallazgo central: no es que RL-RP sea peor, es que es *inestable*

IQR mediano de HVR sobre las 30 semillas:

| balanced | R2-EMOA | d-SMS-EMOA | RL-RP |
|---|---|---|---|
| 0.0067 | 0.0180 | 0.0203 | **0.0909** |

RL-RP es **14 veces más variable** que el punto fijo. En 7 de 14 problemas su
IQR supera 0.1:

```
problema      HVR mediana   IQR
minus-dtlz1      0.7569    0.5530
wfg4             0.6108    0.3918
wfg9             0.9000    0.3793
imop8            0.5525    0.3449
imop6            0.6641    0.3127
dtlz1            0.9689    0.3002
imop5            0.8246    0.1035
```

Nótese dtlz1 (0.9689 frente a 0.9691 de balanced) y wfg9 (0.9000 frente a
0.9065): **en algunas semillas RL-RP iguala al mejor método**, y en otras se
desploma. Es exactamente la firma de congelar una caminata aleatoria en un
instante arbitrario: la traza de una corrida en DTLZ2 alcanza HVR = 0.9408 en
t ≈ 37 800 —**por encima de cualquier baseline**— y termina en 0.5793 porque
`t_adapt = ρ·T_max` cae en un punto malo del recorrido de z_ref.

*Caso aparte:* en DTLZ2 el IQR de RL-RP es 2.7×10⁻⁵, es decir **las 30
semillas convergen al mismo mal valor** (0.5793). Ahí no hay varianza: hay un
atractor determinista. Merece investigarse por separado.

### d-SMS-EMOA: buena trayectoria, mal punto final

Gana 4 de 14 en anytime (D4) pero sólo 2 en HVR final, y es el método que más
se **degrada** (anytime > final en 4 de 14 problemas, hasta +0.106). En DTLZ2,
WFG4 y minus-DTLZ2 su curva anytime empata o supera a la del punto fijo
mientras su valor final queda claramente por debajo.

La explicación es mecánica y estaba predicha por el diseño: el calendario
termina en **r = 1**, exactamente sobre el nadir, donde las soluciones
extremas aportan hipervolumen cero. El barrido de 10 a 1 compra dispersión
temprana y la paga al final. Es un resultado limpio a favor de la tesis: *el
calendario del punto de referencia sí importa para el comportamiento anytime,
pero su punto de llegada decide la calidad final*.

### ⚠ Salvedad sobre WFG4: los números de esa fila hay que recalcularlos

Al regenerar los frentes finales (`dump_fronts.py`, que revalida cada corrida
contra su HVR registrado) **las 6 celdas de WFG4 fallaron la verificación** y
ninguna de las otras 78 falló. Diferencias de 1×10⁻³ a 5×10⁻³, frente a
5×10⁻⁹ en todo lo demás.

**Causa.** pymoo no le da frente analítico a WFG4: cae en
`WFG._calc_pareto_front`, que **aproxima el frente por muestreo aleatorio**
(200 iteraciones × 200 puntos interiores) con una semilla tomada de la
entropía del sistema. Como el experimento construye el marco de referencia
**en cada proceso trabajador por separado**, cada worker usó una Z distinta.
Medido: HV(Z) de WFG4 pasó de 0.75571 a 0.75061 entre dos procesos, un 0.7 %.
WFG9 no se ve afectado porque sí sobreescribe `_calc_pareto_front` con una
versión determinista.

**Alcance.** Sólo WFG4, y sólo en los tres indicadores externos (HVR, IGD+,
Eratio), que se miden contra Z. La búsqueda misma es correcta y reproducible:
las poblaciones finales de WFG4 se regeneran bit a bit. Las diferencias
(≤0.5 %) son dos órdenes de magnitud menores que las que separan a los métodos
en WFG4 (0.9928 / 0.9190 / 0.8742 / 0.6108), así que **el orden de la tabla no
cambia**, pero la precisión reportada para esa fila está sobreestimada y su IQR
está inflado.

**Corregido** en `problems._pinned_pymoo_rng`: el muestreador queda fijado a
una semilla constante, de modo que Z vuelve a ser una propiedad del problema
como exige la sección 2 de `performance.py`. Verificado idéntico en tres
procesos independientes.
`tests_reference_sets.py --determinism` reconstruye cada Z dos veces y falla si
alguna difiere, para que una futura actualización de pymoo no reintroduzca esto
en silencio.

**Pendiente:** volver a correr las 120 celdas de WFG4 con la Z fija. No hace
falta tocar ningún otro problema.

### Dos problemas que rompen el patrón

- **IMOP5** (ocho parches disconexos) es el único donde el punto fijo
  **fracasa**: HVR 0.2818 frente a 0.885 (R2), 0.860 (d-SMS) y 0.825 (RL-RP).
  El único caso del conjunto donde adaptar el PR paga claramente.
- **IMOP7** (banda delgada): los cuatro métodos empatan en HVR = 0.1731.
  Ninguno resuelve el problema y el punto de referencia es irrelevante.

### R2-EMOA confirma la hipótesis pre-registrada de la sección 12

R2-EMOA gana 5 de 14 en HVR e IGD+, y **cuatro de esos cinco son problemas
irregulares o invertidos**: IMOP2, IMOP4, IMOP6 y Minus-DTLZ2 (más IMOP5 como
segundo). Es literalmente lo que la sección 12 anticipó: *"si R2-EMOA gana en
Minus-DTLZ/IMOP, ajustar z_ref es optimizar la perilla equivocada"*. En esas
geometrías el cuello de botella no es dónde está el punto de referencia sino
que la selección por hipervolumen es la herramienta equivocada.

### Lo que RL-RP sí consigue

Gana 3 de 14 en **Eratio** (uniformidad), el término que su recompensa
optimiza explícitamente vía la energía de Riesz. El planeador optimiza lo que
se le pide; el problema es que lo que se le pide no es lo que mide el
desempeño. Consistente con la sección 14: la recompensa no es débil, está
*mal especificada*.

### Siguiente experimento, ahora sí motivado por datos

Congelar z_ref en el **mejor estado visto** durante la adaptación en vez de en
`ρ·T_max`. Toda la evidencia apunta ahí: el planeador visita puntos de
referencia excelentes y no los conserva. Es un argmax sobre una columna que ya
se registra en cada generación, y convierte el IQR de 0.55 en el objeto de
estudio en lugar de en ruido.

---

## 16. Otros bugs corregidos

- `hypervolume` calculaba la máscara de puntos dominantes y **la ignoraba**.
- `select_subaction` usaba `argmax` sobre una fila de ceros → devolvía siempre
  la acción 0. Ahora desempata al azar.
- `sms_emoa_eliminate`: cuando ningún miembro del peor frente domina a z_ref,
  todas las contribuciones son 0 y `argmin` devolvía determinísticamente el
  índice 0 — la selección degeneraba en "borrar siempre al primero". Ahora hay
  criterio secundario.
- `tests_reference_sets.py`: pymoo espera los parámetros posicionales de WFG en
  rango unitario y aplica el escalado por `xu` internamente.
- Documentación de IMOP1/IMOP2 con las geometrías intercambiadas (sección 11).
