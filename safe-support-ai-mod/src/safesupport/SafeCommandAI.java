package safesupport;

import arc.math.geom.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.entities.units.*;

/**
 * CommandAI normal del juego, pero cuando la orden es mover una unidad aerea a una posicion,
 * rodea las zonas de peligro en vez de atravesarlas.
 *
 * Solo cambia el desvio al mover: no toca los ataques a un objetivo, la patrulla, ni las demas ordenes
 * (reparar, reconstruir, asistir y minar siguen usando sus propias IAs seguras).
 * Si el destino esta dentro de una zona, la unidad espera en el borde.
 */
public class SafeCommandAI extends CommandAI{
    static final float waypointReach = 8f, waypointSmooth = 20f;

    protected final SafetyGuard guard = new SafetyGuard();
    private boolean steering;

    /** @return true si esta unidad esta moviendose a una posicion y toca esquivar amenazas. */
    private boolean avoiding(){
        int mode = SafetyGuard.moveAvoid;
        if(mode <= 0 || command != UnitCommand.moveCommand || targetPos == null || attackTarget != null) return false;
        if(hasStance(UnitStance.patrol) || !unit.isFlying()) return false;
        return mode >= 2 || SafeSupportMod.supported(unit);
    }

    @Override
    public void updateUnit(){
        if(avoiding()) guard.update(unit);
        super.updateUnit();
    }

    @Override
    public void moveTo(Position target, float circleLength, float smooth, boolean keepDistance, @Nullable Vec2 offset, boolean arrive){
        if(steering || !avoiding()){
            super.moveTo(target, circleLength, smooth, keepDistance, offset, arrive);
            return;
        }

        //dentro de una zona (por ejemplo por inercia al rodearla): salir primero; si no, el calculo ignora esa zona y cruza recto
        Vec2 esc = guard.escapePoint(unit);
        Vec2 w = esc != null ? esc : guard.waypoint(unit, target);
        steering = true;
        try{
            if(w != null) super.moveTo(w, waypointReach, waypointSmooth);
            else super.moveTo(target, circleLength, smooth, keepDistance, offset, arrive);
        }finally{
            steering = false;
        }
    }
}
