package safesupport;

import arc.math.geom.*;
import arc.struct.*;
import arc.util.*;
import mindustry.entities.*;
import mindustry.gen.*;
import mindustry.world.blocks.defense.turrets.*;
import mindustry.world.meta.*;

import static mindustry.Vars.*;

/**
 * Detecta enemigos y torretas enemigas cerca de una unidad y los convierte en zonas de peligro.
 * Cada unidad de soporte tiene su propia instancia.
 */
public class SafetyGuard{
    /** Radio de busqueda de amenazas alrededor de la unidad. */
    public static float scanRadius = 80f * tilesize;
    /** Margen que se suma al alcance de una unidad enemiga. */
    public static float unitPad = 4f * tilesize;
    /** Margen que se suma al alcance de una torreta enemiga. */
    public static float turretPad = 3f * tilesize;
    /** Radio minimo de peligro de una unidad enemiga (por si su alcance es muy corto). */
    public static float minUnitDanger = 6f * tilesize;
    /** Cada cuantos ticks se vuelve a escanear. */
    public static float scanInterval = 10f;

    private final Zones zones = new Zones();
    private final float[] tmp = new float[2];
    private final Vec2 escape = new Vec2(), waypoint = new Vec2();
    private float lastScan = -1000f;

    /** Reescanea las amenazas si ya toca. Es barato llamarlo en cada frame. */
    public void update(Unit unit){
        float now = Time.time;
        if(now >= lastScan && now - lastScan < scanInterval) return;
        lastScan = now;

        zones.clear();
        final boolean air = unit.isFlying();

        //unidades enemigas que pueden dispararle a esta unidad
        Units.nearbyEnemies(unit.team, unit.x, unit.y, scanRadius, e -> {
            if(!e.isValid() || !e.type.canAttack) return;
            if(air ? !e.type.targetAir : !e.type.targetGround) return;

            float range = e.type.maxRange > 0f ? e.type.maxRange : Math.max(e.type.range, 0f);
            zones.add(e.x, e.y, Math.max(minUnitDanger, range) + unitPad);
        });

        //torretas enemigas con municion que pueden alcanzarla
        Seq<Building> turrets = indexer.getEnemy(unit.team, BlockFlag.turret);
        float limit = scanRadius + 20f * tilesize;
        float limit2 = limit * limit;

        for(int i = 0; i < turrets.size; i++){
            Building b = turrets.get(i);
            if(b == null || !b.isValid()) continue;

            float dx = b.x - unit.x, dy = b.y - unit.y;
            if(dx * dx + dy * dy > limit2) continue;

            if(!(b.block instanceof Turret t)) continue;
            if(!(b instanceof Turret.TurretBuild tb)) continue;
            if(air ? !t.targetAir : !t.targetGround) continue;
            if(!tb.hasAmmo()) continue;

            zones.add(b.x, b.y, tb.range() + turretPad);
        }
    }

    /** @return true si hay alguna amenaza detectada cerca. */
    public boolean hasThreats(){
        return zones.count() > 0;
    }

    /** @return true si el punto (en coordenadas de mundo) cae dentro de una zona de peligro. */
    public boolean dangerAt(float x, float y, float margin){
        return zones.dangerAt(x, y, margin);
    }

    /** @return un punto seguro al que huir si la unidad esta dentro de una zona; null si esta a salvo. */
    public @Nullable Vec2 escapePoint(Unit unit){
        if(!zones.escape(unit.x, unit.y, tmp)) return null;
        return escape.set(tmp[0], tmp[1]);
    }

    /** @return un punto intermedio para rodear las zonas hacia el destino; null si el camino directo es seguro. */
    public @Nullable Vec2 waypoint(Unit unit, Position target){
        if(!zones.waypoint(unit.x, unit.y, target.getX(), target.getY(), tmp)) return null;
        return waypoint.set(tmp[0], tmp[1]);
    }
}
