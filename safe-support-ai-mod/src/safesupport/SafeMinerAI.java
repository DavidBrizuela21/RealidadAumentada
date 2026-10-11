package safesupport;

import arc.math.*;
import arc.math.geom.*;
import arc.util.*;
import mindustry.ai.types.*;
import mindustry.content.*;
import mindustry.gen.*;
import mindustry.type.*;
import mindustry.core.*;
import mindustry.world.*;

import static mindustry.Vars.*;

/**
 * IA de mineria (mono) que no vuela a minar donde hay enemigos o torretas enemigas.
 * Reutiliza MinerAI y le agrega:
 * - sale de inmediato si esta dentro de una zona de peligro;
 * - si la veta elegida esta en una zona de peligro, va a otra veta segura del mismo mineral;
 *   si no hay ninguna, espera junto al nucleo hasta que se despeje;
 * - al volar rodea las zonas de peligro en vez de atravesarlas.
 */
public class SafeMinerAI extends MinerAI{
    static final float waypointReach = 8f, waypointSmooth = 20f, waitDistance = 60f;

    protected final SafetyGuard guard = new SafetyGuard();
    private boolean steering;
    private Tile altOre;
    private Item altItem;
    private float lastAltSearch = -1000f;

    @Override
    public void updateMovement(){
        guard.update(unit);

        //dentro de una zona de peligro: salir de inmediato
        Vec2 esc = guard.escapePoint(unit);
        if(esc != null){
            unit.mineTile(null);
            rawMoveTo(esc, waypointReach);
            return;
        }

        //la veta esta en una zona de peligro: ir a otra veta segura del mismo mineral o esperar junto al nucleo
        float margin = unit.type.mineRange / 2f; //la unidad se queda a esa distancia de la veta
        if(mining && ore != null && guard.dangerAt(ore.worldx(), ore.worldy(), margin)){
            Tile alt = safeOre(margin);

            if(alt != null){
                ore = alt;
            }else{
                unit.mineTile(null);

                Building core = unit.closestCore();
                if(core != null && !unit.within(core, waitDistance * 2f)){
                    rawMoveTo(core, waitDistance);
                }
                return;
            }
        }

        //de regreso al nucleo con carga: MinerAI usa circle() y no moveTo(), asi que se rodea aqui
        if(!mining){
            Building core = unit.closestCore();
            if(core != null){
                Vec2 w = guard.waypoint(unit, core);
                if(w != null){
                    unit.mineTile(null);
                    rawMoveTo(w, waypointReach);
                    return;
                }
            }
        }

        super.updateMovement();

        //MinerAI vuelve a elegir la veta mas cercana al nucleo cada 4 s: si es la peligrosa, no la mina
        if(mining && ore != null && guard.dangerAt(ore.worldx(), ore.worldy(), margin)){
            unit.mineTile(null);
        }
    }

    /** La veta segura mas cercana al nucleo del mineral actual. Se busca como mucho una vez por segundo. */
    private @Nullable Tile safeOre(float margin){
        Building core = unit.closestCore();
        Item item = targetItem;
        if(core == null || item == null) return null;

        if(altOre != null && altItem == item && !guard.dangerAt(altOre.worldx(), altOre.worldy(), margin)) return altOre;

        float now = Time.time;
        if(now >= lastAltSearch && now - lastAltSearch < 60f) return null;
        lastAltSearch = now;
        altItem = item;
        altOre = null;

        int cx = World.toTile(core.x), cy = World.toTile(core.y), r = (int)(SafetyGuard.oreSearch / tilesize);
        float bestDst = Float.MAX_VALUE;

        for(int x = cx - r; x <= cx + r; x++){
            for(int y = cy - r; y <= cy + r; y++){
                Tile t = world.tile(x, y);
                if(t == null) continue;

                boolean floorOre = unit.type.mineFloor && t.drop() == item && t.block() == Blocks.air;
                boolean wallOre = unit.type.mineWalls && t.wallDrop() == item && t.block() != Blocks.air;
                if(!floorOre && !wallOre) continue;

                float dst = Mathf.dst2(t.worldx(), t.worldy(), core.x, core.y);
                if(dst >= bestDst || guard.dangerAt(t.worldx(), t.worldy(), margin)) continue;

                altOre = t;
                bestDst = dst;
            }
        }

        return altOre;
    }

    /** Mueve a la unidad sin aplicar la esquiva (para huir o ir a un punto intermedio). */
    private void rawMoveTo(Position p, float circleLength){
        boolean prev = steering;
        steering = true;
        try{
            moveTo(p, circleLength);
        }finally{
            steering = prev;
        }
    }

    //---- esquiva: todas las variantes de moveTo pasan por el mismo desvio ----

    @Override
    public void moveTo(Position target, float circleLength){
        if(steering){
            super.moveTo(target, circleLength);
            return;
        }

        Vec2 w = guard.waypoint(unit, target);
        steering = true;
        try{
            if(w != null) super.moveTo(w, waypointReach, waypointSmooth);
            else super.moveTo(target, circleLength);
        }finally{
            steering = false;
        }
    }

    @Override
    public void moveTo(Position target, float circleLength, float smooth){
        if(steering){
            super.moveTo(target, circleLength, smooth);
            return;
        }

        Vec2 w = guard.waypoint(unit, target);
        steering = true;
        try{
            if(w != null) super.moveTo(w, waypointReach, waypointSmooth);
            else super.moveTo(target, circleLength, smooth);
        }finally{
            steering = false;
        }
    }

    @Override
    public void moveTo(Position target, float circleLength, float smooth, boolean keepDistance, @Nullable Vec2 offset){
        if(steering){
            super.moveTo(target, circleLength, smooth, keepDistance, offset);
            return;
        }

        Vec2 w = guard.waypoint(unit, target);
        steering = true;
        try{
            if(w != null) super.moveTo(w, waypointReach, waypointSmooth);
            else super.moveTo(target, circleLength, smooth, keepDistance, offset);
        }finally{
            steering = false;
        }
    }

    @Override
    public void moveTo(Position target, float circleLength, float smooth, boolean keepDistance, @Nullable Vec2 offset, boolean arrive){
        if(steering){
            super.moveTo(target, circleLength, smooth, keepDistance, offset, arrive);
            return;
        }

        Vec2 w = guard.waypoint(unit, target);
        steering = true;
        try{
            if(w != null) super.moveTo(w, waypointReach, waypointSmooth);
            else super.moveTo(target, circleLength, smooth, keepDistance, offset, arrive);
        }finally{
            steering = false;
        }
    }
}
