package safesupport;

import arc.math.*;
import arc.math.geom.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.entities.units.*;
import mindustry.gen.*;
import mindustry.world.blocks.ConstructBlock.*;

import static mindustry.Vars.*;

/**
 * IA de reparacion (mega) que no se lanza a reparar donde hay enemigos o torretas enemigas.
 * Basada en la logica de RepairAI del juego, con estas diferencias:
 * - ignora los bloques danados que estan dentro de una zona de peligro (espera a que se despeje);
 * - si la unidad se encuentra dentro de una zona de peligro, sale de inmediato;
 * - al volar rodea las zonas de peligro en vez de atravesarlas.
 */
public class SafeRepairAI extends AIController{
    public static float retreatDst = 160f, fleeRange = 310f, retreatDelay = 60f * 3f;
    static final float waypointReach = 8f, waypointSmooth = 20f;

    protected final SafetyGuard guard = new SafetyGuard();
    @Nullable Teamc avoid;
    float retreatTimer;
    Building damagedTarget;
    private boolean steering;
    /** Esta protegida por un muro: se queda quieta y solo cura lo que alcanza. */
    private boolean inCover;
    /** Bloque danado dentro de una zona de peligro que se puede curar desde un punto cubierto; solo si no hay nada seguro que curar. */
    private Building healAnchor;
    /** Fraccion del alcance a la que se busca el punto cubierto (deja margen para que el proyectil llegue). */
    static final float anchorReach = 0.8f;

    @Override
    public void updateMovement(){
        guard.update(unit);

        boolean hold = hasStance(UnitStance.holdPosition);

        //dentro de una zona de peligro: cubrirse tras un muro si hay uno, si no salir de inmediato
        Vec2 esc = guard.escapePoint(unit);
        inCover = false;
        if(esc != null && !hold){
            retreatTimer = 0f;

            if(guard.covered(unit)){
                inCover = true;

                //nada que curar desde aqui: pasar a un punto cubierto mas cerca del bloque danado
                if(!holdInCover()){
                    Vec2 np = healCover();
                    if(np != null && Mathf.dst(unit.x, unit.y, np.x, np.y) > 12f) rawMoveTo(np, waypointReach);
                }
                return;
            }

            //con un bloque danado que curar, el punto cubierto cercano a el; si no, el muro mas cercano
            Vec2 dest = healCover();
            if(dest == null){
                Vec2 cp = guard.coverPoint(unit);
                dest = cp != null && Mathf.dst(unit.x, unit.y, cp.x, cp.y) <= Mathf.dst(unit.x, unit.y, esc.x, esc.y) + SafetyGuard.coverDetour ? cp : esc;
            }

            unit.controlWeapons(false);
            damagedTarget = null;
            target = null;
            rawMoveTo(dest, waypointReach);
            return;
        }

        //el objetivo quedo en una zona de peligro: soltarlo
        if(target instanceof Building tb && tb.team == unit.team && guard.dangerAt(tb.x, tb.y, 0f)){
            target = null;
            damagedTarget = null;
        }

        //nada seguro que curar, pero hay un bloque danado en una zona de peligro
        if(!hold && !(target instanceof Building) && healAnchor != null && healAnchor.isValid() && unit.healthf() > 0.5f){
            if(healAnchor.within(unit, unit.type.range * anchorReach)){
                //lo alcanza desde fuera de la zona
                retreatTimer = 0f;
                unit.aim(healAnchor);
                unit.lookAt(healAnchor);
                unit.controlWeapons(true);
                return;
            }

            Vec2 cp = guard.coverNear(unit, healAnchor.x, healAnchor.y, unit.type.range * anchorReach);
            if(cp != null){
                retreatTimer = 0f;
                unit.controlWeapons(false);
                rawMoveTo(cp, waypointReach);
                return;
            }
        }

        if(target instanceof Building){
            boolean shoot = false;

            if(target.within(unit, unit.type.range)){
                unit.aim(target);
                shoot = true;
            }

            unit.controlWeapons(shoot);
        }else if(target == null){
            unit.controlWeapons(false);
        }

        if(target != null && target instanceof Building b && b.team == unit.team){
            if(!hold){
                if(unit.type.circleTarget){
                    circleAttack(unit.type.circleTargetRadius);
                }else if(!target.within(unit, unit.type.range * 0.65f)){
                    moveTo(target, unit.type.range * 0.65f);
                }
            }

            if(!unit.type.circleTarget){
                unit.lookAt(target);
            }
        }

        //sin nada que reparar
        if(!(target instanceof Building) && !hold){
            if(timer.get(timerTarget4, 40)){
                avoid = target(unit.x, unit.y, fleeRange, true, true);
            }

            if((retreatTimer += Time.delta) >= retreatDelay){
                //alejarse del enemigo cuando no hay nada que hacer
                if(avoid != null){
                    var core = unit.closestCore();
                    if(core != null && !unit.within(core, retreatDst)){
                        moveTo(core, retreatDst);
                    }
                }
            }
        }else{
            retreatTimer = 0f;
        }
    }

    @Override
    public void updateTargeting(){
        guard.update(unit);

        if(timer.get(timerTarget, 15)) scanDamaged();

        if(damagedTarget != null && !damagedTarget.isValid()) damagedTarget = null;

        if(damagedTarget == null){
            super.updateTargeting();
        }else{
            this.target = damagedTarget;
        }
    }

    /** Punto cubierto cerca del bloque danado en zona de peligro, si esta a menos de coverSearch de la unidad (para no cruzar medio mapa bajo fuego). */
    private @Nullable Vec2 healCover(){
        if(healAnchor == null || !healAnchor.isValid()) return null;

        Vec2 np = guard.coverNear(unit, healAnchor.x, healAnchor.y, unit.type.range * anchorReach);
        return np != null && Mathf.dst(unit.x, unit.y, np.x, np.y) <= SafetyGuard.coverSearch ? np : null;
    }

    /**
     * Protegida por un muro: no se mueve, apunta al bloque danado mas cercano que alcanza y lo cura.
     * @return true si esta curando algo
     */
    private boolean holdInCover(){
        if(target instanceof Building b && b.team == unit.team && target.within(unit, unit.type.range)){
            unit.aim(target);
            unit.lookAt(target);
            unit.controlWeapons(true);
            return true;
        }

        unit.controlWeapons(false);
        return false;
    }

    /**
     * Recorre los bloques danados una sola vez y decide:
     * - objetivo normal: el mas cercano fuera de las zonas de peligro (o, cubierta, el mas cercano al alcance);
     * - ancla: si no hay objetivo, el mas cercano dentro de una zona, para curarlo desde un punto cubierto.
     */
    private void scanDamaged(){
        var list = indexer.getDamaged(unit.team);
        float reach = unit.type.range * 0.9f;
        Building safe = null, inRange = null, danger = null;
        float safeDst = Float.MAX_VALUE, rangeDst = reach * reach, dangerDst = Float.MAX_VALUE;

        for(int i = 0; i < list.size; i++){
            Building b = list.get(i);
            if(b == null || !b.isValid() || b instanceof ConstructBuild) continue;

            float dst = b.dst2(unit.x, unit.y);
            if(guard.dangerAt(b.x, b.y, 0f)){
                if(dst < dangerDst){
                    danger = b;
                    dangerDst = dst;
                }
            }else if(dst < safeDst){
                safe = b;
                safeDst = dst;
            }

            if(dst < rangeDst){
                inRange = b;
                rangeDst = dst;
            }
        }

        damagedTarget = inCover ? inRange : safe;
        healAnchor = damagedTarget == null ? danger : null;
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
