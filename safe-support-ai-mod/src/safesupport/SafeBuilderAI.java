package safesupport;

import arc.math.geom.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.entities.units.*;

import static mindustry.Vars.*;

/**
 * IA de reconstruccion/asistencia (poly) que no se lanza a construir donde hay enemigos o torretas enemigas.
 * Reutiliza BuilderAI y le agrega:
 * - sale de inmediato si esta dentro de una zona de peligro;
 * - descarta los planes de construccion que estan dentro de una zona de peligro (los retoma cuando se despeje);
 * - al volar rodea las zonas de peligro en vez de atravesarlas.
 */
public class SafeBuilderAI extends BuilderAI{
    static final float waypointReach = 8f, waypointSmooth = 20f;

    protected final SafetyGuard guard = new SafetyGuard();
    private boolean steering;

    @Override
    public void updateMovement(){
        guard.update(unit);

        boolean hold = hasStance(UnitStance.holdPosition);

        //dentro de una zona de peligro: dejar de construir y salir
        Vec2 esc = guard.escapePoint(unit);
        if(esc != null && !hold){
            unit.clearBuilding();
            following = null;
            lastPlan = null;
            rawMoveTo(esc, waypointReach);
            return;
        }

        //el plan actual esta en una zona de peligro: descartarlo (el plan sigue en la cola del equipo)
        BuildPlan plan = unit.buildPlan();
        if(plan != null && guard.dangerAt(plan.x * tilesize, plan.y * tilesize, 0f)){
            unit.plans.removeFirst();
            following = null;
            lastPlan = null;
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
