package safesupport;

import arc.*;
import arc.func.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.entities.units.*;
import mindustry.game.*;
import mindustry.gen.*;
import mindustry.mod.*;

import java.lang.reflect.*;

/**
 * Cambia la IA de los comandos de soporte de Serpulo (reparar, reconstruir, asistir, minar)
 * por versiones que evitan enemigos y torretas enemigas.
 *
 * Los comandos son objetos globales de UnitCommand cuyo campo "controller" es final,
 * asi que se reemplaza por reflexion. Se conservan los comandos originales: menus, atajos
 * y partidas guardadas siguen funcionando igual.
 */
public class SafeSupportMod extends Mod{
    private static boolean patched;

    public SafeSupportMod(){
        //por si los comandos aun no estaban cargados cuando se ejecuta init()
        Events.on(EventType.ClientLoadEvent.class, e -> patch());
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
            setController(UnitCommand.repairCommand, u -> new SafeRepairAI()) &
            setController(UnitCommand.rebuildCommand, u -> new SafeBuilderAI()) &
            setController(UnitCommand.assistCommand, u -> {
                SafeBuilderAI ai = new SafeBuilderAI();
                ai.onlyAssist = true;
                return ai;
            }) &
            setController(UnitCommand.mineCommand, u -> new SafeMinerAI());

        patched = true;
        Log.info("[Safe Support AI] " + (ok ? "IA de soporte segura activada." : "algunos comandos no se pudieron reemplazar, revisa el log."));
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
