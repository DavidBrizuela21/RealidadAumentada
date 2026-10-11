# Safe Support AI (mod de Mindustry)

Mejora la IA de las unidades de soporte de Serpulo (mono, poly, mega).

## Qué hace

| Problema | Comportamiento nuevo |
|---|---|
| Van a reparar/reconstruir aunque haya un enemigo o torreta enemiga al lado | Ignoran los bloques dañados y los planes de construcción que estén en una zona de peligro. Los retoman cuando se despeja |
| Esperan a estar ociosas para huir | Si entran en una zona de peligro, salen de inmediato y sueltan el objetivo |
| Atraviesan torretas enemigas al volar a minar o a reparar | Rodean las zonas de peligro en vez de cruzarlas |
| El mono vuela a minar donde hay enemigos | Si la veta está en una zona de peligro, busca otra veta segura del mismo mineral (hasta 40 casillas del núcleo); si no hay ninguna, espera junto al núcleo. Al volver a soltar la carga también rodea |
| Huir siempre aunque haya un muro que proteja | **Poly y mega** se cubren detrás de un muro propio (categoría muros: cobre, titanio, etc.) en vez de huir. El mega se queda quieto y cura lo que alcanza; el poly se queda quieto sin construir |

**Zona de peligro** = alcance de cada unidad enemiga que pueda disparar a la unidad (aérea o terrestre según el caso) + 4 casillas, y alcance de cada torreta enemiga con munición que pueda alcanzarla + 3 casillas.

**Cobertura.** Una unidad dentro de una zona de peligro se considera cubierta si, desde *cada* amenaza que la alcanza, hay un muro propio en la línea de tiro (se comprueban tres líneas paralelas para cubrir el ancho de la unidad) y las balas de esa amenaza las frena un muro. No cuentan como cubierta las amenazas con láseres, rayos continuos, balas perforantes o metralla antiaérea (Scatter), que atraviesan muros. Si no hay cobertura, busca el muro más cercano (hasta 10 casillas) y se pone 1,5 casillas detrás; si no hay ninguno, huye como antes. Puertas y bloques que las balas sobrevuelan (cintas, puentes, conductos) no cuentan como muro.

Se aplica a los comandos de siempre (reparar, reconstruir, asistir, minar), así que los menús, atajos y partidas guardadas no cambian. Respeta la postura "mantener posición".

## Estructura

```
mod.hjson
src/safesupport/
  SafeSupportMod.java   reemplaza la IA de los 4 comandos al cargar
  SafetyGuard.java      detecta enemigos y torretas, arma las zonas de peligro
  Zones.java            geometría: rodear, esperar en el borde, escapar (sin dependencias, probada)
  Cover.java            geometría de cobertura: líneas de tiro y muros (sin dependencias, probada)
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

En el juego: **Configuración → Safe Support AI**. Los cambios se aplican en un segundo, sin reiniciar.

| Ajuste | Por defecto | Efecto |
|---|---|---|
| Radio de búsqueda de amenazas | 80 casillas | Hasta dónde busca enemigos y torretas |
| Margen sobre el alcance de enemigos | 4 casillas | Se suma al alcance de una unidad enemiga |
| Margen sobre el alcance de torretas | 3 casillas | Se suma al alcance de una torreta enemiga |
| Peligro mínimo de un enemigo | 6 casillas | Radio mínimo de peligro, aunque su alcance sea corto |
| Búsqueda de muros para cubrirse | 10 casillas | Hasta dónde busca un muro propio (poly y mega) |
| Distancia detrás del muro | 1,5 casillas | Entre el borde del muro y la unidad (las balas que explotan hacen daño en área) |
| Búsqueda de veta alternativa | 40 casillas | Hasta dónde busca el mono otra veta, desde el núcleo |
| Mostrar zonas de peligro | apagado | Modo diagnóstico (abajo) |

Las constantes `scanInterval`, `coverInterval`, `coverChecks`, `coverSpread` y `coverDetour` siguen en `SafetyGuard.java`.

### Modo diagnóstico

Con **Mostrar zonas de peligro** activado se dibuja sobre el mapa, para las unidades seleccionadas (o las primeras 8 si no hay ninguna seleccionada):

- **Círculo naranja:** amenaza cuyas balas frena un muro.
- **Círculo rojo:** amenaza que atraviesa muros (láser, metralla antiaérea, balas perforantes) o de la que no se pudo saber.
- **Anillo verde en la unidad:** está dentro de una zona pero cubierta por un muro.
- **Anillo rojo en la unidad:** está dentro de una zona y expuesta.
- **Punto y línea verdes:** muro elegido para cubrirse.

Si una torreta no aparece dibujada, el mod no la ve como amenaza (sin munición, no puede apuntar a la unidad, o es de tu equipo).

## Verificación realizada

- `Zones` (la geometría) tiene pruebas que se ejecutaron: nunca entra en una zona, rodea una o varias en fila, espera en el borde si el destino está en peligro y escapa si ya está dentro.
- El resto del código compila contra un modelo simplificado de la API de Mindustry y el reemplazo por reflexión funciona en ese modelo.
- **No** se compiló contra el juego real ni se probó en una partida: hay que hacerlo (ver pasos de arriba).

## Limitaciones conocidas

- Solo esquiva amenazas; no coordina qué unidad repara qué bloque.
- El mega y el poly vuelan hasta el borde de la zona y esperan ahí: si hay combate constante cerca de un bloque dañado, no se repara hasta que termine.
- Detrás de un muro el mega cura lo que alcanza, pero su proyectil curativo choca con el primer edificio propio que encuentra: cura sobre todo al propio muro y a lo pegado a él. El poly no tiene curación automática de bloques en el juego, así que solo espera sin reconstruir.
- Una amenaza cuyo tipo de bala no se puede determinar se trata como que atraviesa muros (no hay cobertura).
- La cobertura no se probó en una partida real: la geometría tiene pruebas, el comportamiento en el juego hay que verlo.
- Los comandos (reparar, reconstruir, asistir, minar) son globales, pero la IA segura solo se asigna a mono, poly y mega. Cualquier otra unidad que use esos comandos (por ejemplo las de Erekir) conserva la IA original del juego.
- Escrito contra el código fuente actual del juego (v8, build 15x/16x). Una actualización con cambios en `UnitCommand` o `AIController` puede requerir ajustes.
