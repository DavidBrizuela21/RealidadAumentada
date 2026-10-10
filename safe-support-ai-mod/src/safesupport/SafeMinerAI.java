package safesupport;

import arc.math.geom.*;
import arc.util.*;
import mindustry.ai.types.*;
import mindustry.gen.*;

/**
 * IA de mineria (mono) que no vuela a minar donde hay enemigos o torretas enemigas.
 * Reutiliza MinerAI y le agrega:
 * - sale de inmediato si esta dentro de una zona de peligro;
 * - si la veta elegida esta en una zona de peligro, espera junto al nucleo hasta que se despeje;
 * - al volar rodea las zonas de peligro en vez de atravesarlas.
 */
public class SafeMinerAI extends MinerAI{
    static final float waypointReach = 8f, waypointSmooth = 20f, waitDistance = 60f;

    protected final SafetyGuard guard = new SafetyGuard();
    private boolean steering;

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

        //la veta esta en una zona de peligro: esperar junto al nucleo
        if(mining && ore != null && guard.dangerAt(ore.worldx(), ore.worldy(), 0f)){
            unit.mineTile(null);

            Building core = unit.closestCore();
            if(core != null && !unit.within(core, waitDistance * 2f)){
                rawMoveTo(core, waitDistance);
            }
            return;
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
