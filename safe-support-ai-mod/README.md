# Safe Support AI (mod de Mindustry)

Mejora la IA de las unidades de soporte de Serpulo (mono, poly, mega).

## Qué hace

| Problema | Comportamiento nuevo |
|---|---|
| Van a reparar/reconstruir aunque haya un enemigo o torreta enemiga al lado | Ignoran los bloques dañados y los planes de construcción que estén en una zona de peligro. Los retoman cuando se despeja |
| Esperan a estar ociosas para huir | Si entran en una zona de peligro, salen de inmediato y sueltan el objetivo |
| Atraviesan torretas enemigas al volar a minar o a reparar | Rodean las zonas de peligro en vez de cruzarlas |
| El mono vuela a minar donde hay enemigos | Si la veta está en una zona de peligro espera junto al núcleo. Al volver a soltar la carga también rodea |

**Zona de peligro** = alcance de cada unidad enemiga que pueda disparar a la unidad (aérea o terrestre según el caso) + 4 casillas, y alcance de cada torreta enemiga con munición que pueda alcanzarla + 3 casillas.

Se aplica a los comandos de siempre (reparar, reconstruir, asistir, minar), así que los menús, atajos y partidas guardadas no cambian. Respeta la postura "mantener posición".

## Estructura

```
mod.hjson
src/safesupport/
  SafeSupportMod.java   reemplaza la IA de los 4 comandos al cargar
  SafetyGuard.java      detecta enemigos y torretas, arma las zonas de peligro
  Zones.java            geometría: rodear, esperar en el borde, escapar (sin dependencias, probada)
  SafeRepairAI.java     mega
  SafeBuilderAI.java    poly
  SafeMinerAI.java      mono
```

## Compilar (sin instalar nada) con GitHub Actions

1. En GitHub abre https://github.com/Anuken/MindustryJavaModTemplate y pulsa **Use this template** para crear tu propio repositorio.
2. En tu repositorio: borra la carpeta `src/example`, sube la carpeta `src/safesupport` y reemplaza `mod.hjson` por el de este proyecto.
3. Al hacer el commit se ejecuta **Actions**. Cuando termine, descarga el artefacto desde la pestaña Actions (llega como zip, descomprímelo para obtener el `.jar`).
4. Ese `.jar` sirve en Android y en PC.

## Instalar en Android

Mindustry → **Mods** → **Import Mod** → elige el `.jar`. Reinicia el juego. En el log debe aparecer `[Safe Support AI] IA de soporte segura activada.`

## Ajustes

En `SafetyGuard.java` (valores en unidades de mundo; 1 casilla = 8):

| Constante | Valor | Efecto |
|---|---|---|
| `scanRadius` | 80 casillas | Hasta dónde busca amenazas |
| `unitPad` | 4 casillas | Margen extra sobre el alcance de un enemigo |
| `turretPad` | 3 casillas | Margen extra sobre el alcance de una torreta |
| `minUnitDanger` | 6 casillas | Radio mínimo de peligro de una unidad enemiga |
| `scanInterval` | 10 ticks | Cada cuánto reescanea |

## Verificación realizada

- `Zones` (la geometría) tiene pruebas que se ejecutaron: nunca entra en una zona, rodea una o varias en fila, espera en el borde si el destino está en peligro y escapa si ya está dentro.
- El resto del código compila contra un modelo simplificado de la API de Mindustry y el reemplazo por reflexión funciona en ese modelo.
- **No** se compiló contra el juego real ni se probó en una partida: hay que hacerlo (ver pasos de arriba).

## Limitaciones conocidas

- Solo esquiva amenazas; no coordina qué unidad repara qué bloque.
- El mega y el poly vuelan hasta el borde de la zona y esperan ahí: si hay combate constante cerca de un bloque dañado, no se repara hasta que termine.
- Los comandos son globales: también afectan a otras unidades que usen reparar/reconstruir/asistir/minar (por ejemplo en Erekir).
- Escrito contra el código fuente actual del juego (v8, build 15x/16x). Una actualización con cambios en `UnitCommand` o `AIController` puede requerir ajustes.
