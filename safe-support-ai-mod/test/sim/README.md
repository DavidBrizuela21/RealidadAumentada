# Simulacion con el juego real

`Sim.java` (reparar, reconstruir, minar) y `SimMove.java` (comando mover) levantan Mindustry sin pantalla con un mapa
minimo, una o varias torretas enemigas con municion y una unidad de soporte, y miden cuanto se acerca la unidad al
alcance de las torretas. No forman parte del mod.

Necesitan los mapas por defecto del juego (assets que no vienen en el jar):

    git clone --depth 1 --branch v160 --filter=blob:none --sparse https://github.com/Anuken/Mindustry.git mrepo
    (cd mrepo && git sparse-checkout set core/assets/maps core/assets/bundles)

Compilar y ejecutar (J = jar de Mindustry de la cache de Gradle, M = clases del mod compiladas):

    javac -proc:none -sourcepath "" -implicit:none -cp $J:$M -d . Sim.java SimMove.java
    java -Dcmd=1 -cp $J:$M:.:mrepo/core/assets Sim mega duo           # IA de reparar con orden real
    java -Dpatch=1 -Dwall=1 -Dh=140 -cp $J:$M:.:mrepo/core/assets SimMove zenith duo   # mover con una pared de 5 torretas

Opciones: `-Dpatch=1` aplica el parche del mod (sin ella es el juego sin mod, para comparar), `-Dwall=1` pared de
torretas, `-Dh=N` alto del mapa en casillas, `-Dcluster=1` varias torretas, `-Ddelta=N` tiempo por cuadro, `-Dtrace=1`
trayectoria.
