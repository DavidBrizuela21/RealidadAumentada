package safesupport;

import arc.math.*;
import arc.math.geom.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.entities.units.*;
import mindustry.gen.*;

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
    private Building healTarget;
    private float healTimer;

    @Override
    public void updateMovement(){
        guard.update(unit);

        boolean hold = hasStance(UnitStance.holdPosition);

        //dentro de una zona de peligro: dejar de construir y cubrirse tras un muro si hay uno, si no salir
        Vec2 esc = guard.escapePoint(unit);
        if(esc != null && !hold){
            unit.clearBuilding();
            following = null;
            lastPlan = null;

            //protegida por un muro: se queda quieta, cura lo que alcanza y retoma el trabajo cuando se despeje
            if(guard.covered(unit)){
                healFromCover();
                return;
            }

            Vec2 cp = guard.coverPoint(unit);
            Vec2 dest = cp != null && Mathf.dst(unit.x, unit.y, cp.x, cp.y) <= Mathf.dst(unit.x, unit.y, esc.x, esc.y) + SafetyGuard.coverDetour ? cp : esc;
            rawMoveTo(dest, waypointReach);
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

    /** Sus misiles curan bloques propios (healPercent), pero la IA solo apunta a enemigos: desde cobertura apunta al bloque danado mas cercano. */
    private void healFromCover(){
        if((healTimer += Time.delta) >= 15f || (healTarget != null && !healTarget.isValid())){
            healTimer = 0f;
            healTarget = SafetyGuard.damagedNear(unit, unit.type.range * 0.9f);
        }

        if(healTarget != null && healTarget.isValid() && healTarget.within(unit, unit.type.range)){
            unit.aim(healTarget);
            unit.lookAt(healTarget);
            unit.controlWeapons(true);
        }else{
            unit.controlWeapons(false);
        }
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
