package safesupport;

import arc.*;
import arc.func.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.content.*;
import mindustry.entities.units.*;
import mindustry.game.*;
import mindustry.gen.*;
import mindustry.mod.*;

import java.lang.reflect.*;

/**
 * Cambia la IA de los comandos de soporte de Serpulo (reparar, reconstruir, asistir, minar)
 * por versiones que evitan enemigos y torretas enemigas.
 *
 * Solo las unidades de soporte de Serpulo (mono, poly, mega) usan las versiones seguras; las demas
 * conservan la IA original del juego aunque compartan el comando.
 *
 * Los comandos son objetos globales de UnitCommand cuyo campo "controller" es final,
 * asi que se reemplaza por reflexion. Se conservan los comandos originales: menus, atajos
 * y partidas guardadas siguen funcionando igual.
 */
public class SafeSupportMod extends Mod{
    private static boolean patched;

    public SafeSupportMod(){
        //por si los comandos aun no estaban cargados cuando se ejecuta init()
        Events.on(EventType.ClientLoadEvent.class, e -> {
            patch();
            SafeSettings.register();
        });
        Events.run(EventType.Trigger.drawOver, SafetyGuard::drawDebug);
    }

    @Override
    public void init(){
        patch();
    }

    static void patch(){
        if(patched) return;

        if(UnitCommand.repairCommand == null || UnitCommand.rebuildCommand == null
        || UnitCommand.assistCommand == null || UnitCommand.mineCommand == null){
            return;
        }

        boolean ok =
            setController(UnitCommand.repairCommand, u -> supported(u) ? new SafeRepairAI() : new RepairAI()) &
            setController(UnitCommand.rebuildCommand, u -> supported(u) ? new SafeBuilderAI() : new BuilderAI()) &
            setController(UnitCommand.assistCommand, u -> {
                BuilderAI ai = supported(u) ? new SafeBuilderAI() : new BuilderAI();
                ai.onlyAssist = true;
                return ai;
            }) &
            setController(UnitCommand.mineCommand, u -> supported(u) ? new SafeMinerAI() : new MinerAI());

        patched = true;
        Log.info("[Safe Support AI] " + (ok ? "IA de soporte segura activada." : "algunos comandos no se pudieron reemplazar, revisa el log."));
    }

    /** Solo mono, poly y mega usan la IA segura; el resto de unidades (por ejemplo las de Erekir) conservan la del juego. */
    static boolean supported(Unit unit){
        return unit.type == UnitTypes.mono || unit.type == UnitTypes.poly || unit.type == UnitTypes.mega;
    }

    private static boolean setController(UnitCommand command, Func<Unit, AIController> controller){
        try{
            Field field = UnitCommand.class.getDeclaredField("controller");
            field.setAccessible(true);
            field.set(command, controller);
            return true;
        }catch(Throwable t){
            Log.err("[Safe Support AI] no se pudo reemplazar el controlador de " + command.name, t);
            return false;
        }
    }
}
