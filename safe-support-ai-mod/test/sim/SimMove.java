import arc.*;
import mindustry.*;
import arc.backend.headless.*;
import arc.math.*;
import arc.math.geom.*;
import arc.struct.*;
import arc.util.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.content.*;
import mindustry.core.*;
import mindustry.game.*;
import mindustry.gen.*;
import mindustry.net.*;
import mindustry.type.*;
import mindustry.world.*;
import mindustry.world.blocks.defense.turrets.*;
import safesupport.*;

import static mindustry.Vars.*;

public class SimMove{
    static boolean ready;
    static Throwable err;

    public static void main(String[] args) throws Exception{
        Log.useColors = false;
        new HeadlessApplication(new ApplicationCore(){
            @Override public void setup(){
                headless = true; net = new Net(null); tree = new FileTree(); Vars.init();
                world = new World(){ public float getDarkness(int x, int y){ return 0; } };
                content.createBaseContent(); add(logic = new Logic()); add(netServer = new NetServer()); content.init();
            }
            @Override public void init(){ super.init(); ready = true; }
        }, t -> err = t);
        long t0 = System.currentTimeMillis();
        while(!ready && err == null && System.currentTimeMillis() - t0 < 30000) Thread.sleep(50);
        if(err != null){ err.printStackTrace(); System.exit(2); }
        Time.setDeltaProvider(() -> 1f);
        final String[] fa = args;
        final boolean[] done = {false};
        Core.app.post(() -> { try{ run(fa[0], fa.length > 1 ? fa[1] : "duo"); }catch(Throwable t){ t.printStackTrace(); } done[0] = true; });
        while(!done[0]) Thread.sleep(20);
        System.exit(0);
    }

    static void run(String unitName, String turretName) throws Exception{
        int W = 140, H = Integer.getInteger("h", 80);
        logic.reset();
        world.beginMapLoad();
        Tiles tiles = world.resize(W, H);
        for(int x = 0; x < W; x++) for(int y = 0; y < H; y++) tiles.set(x, y, new Tile(x, y, Blocks.stone, Blocks.air, Blocks.air));
        world.endMapLoad();
        state.rules.waves = false;
        tiles.getn(10, 40).setBlock(Blocks.coreShard, Team.sharded, 0);

        Block tb = content.block(turretName);
        Seq<Building> turrets = new Seq<>();
        int[][] pos = System.getProperty("wall") != null
            ? new int[][]{{60,40},{60,32},{60,48},{60,24},{60,56}}   //pared de torretas
            : new int[][]{{60,40}};
        for(int[] p : pos){
            Tile t = tiles.getn(p[0], p[1]);
            t.setBlock(tb, Team.crux, 0);
            if(t.build instanceof ItemTurret.ItemTurretBuild ib){ for(Item it : ((ItemTurret)tb).ammoTypes.keys()){ for(int i = 0; i < 20; i++) ib.handleItem(ib, it); break; } }
            turrets.add(t.build);
        }

        boolean patch = System.getProperty("patch") != null;
        if(patch){ var pm = SafeSupportMod.class.getDeclaredMethod("patch"); pm.setAccessible(true); pm.invoke(null); }
        state.set(GameState.State.playing);

        UnitType type = content.unit(unitName);
        Unit u = type.spawn(Team.sharded, 14 * 8, 40 * 8);
        u.controller(type.createController(u));
        CommandAI ai = (CommandAI)u.controller();
        ai.command(UnitCommand.moveCommand);
        float destX = Integer.parseInt(System.getProperty("destx", "100")) * 8f, destY = 40 * 8f;
        ai.commandPosition(new Vec2(destX, destY));

        float startHp = u.health, minMargin = Float.MAX_VALUE;
        int arrived = -1;
        for(int tick = 0; tick < 4000; tick++){
            logic.update();
            if(!u.isValid()){ System.out.println("  UNIT DIED at tick " + tick); break; }
            for(Building b : turrets){
                float mg = Mathf.dst(u.x, u.y, b.x, b.y) - ((Turret.TurretBuild)b).range();
                if(mg < minMargin) minMargin = mg;
            }
            if(arrived < 0 && Mathf.dst(u.x, u.y, destX, destY) < 12f) arrived = tick;
            if(System.getProperty("trace") != null && tick % 15 == 0 && tick < 700){
                Building b0 = turrets.first();
                System.out.printf("  t=%d pos=(%.1f,%.1f) dstTurret=%.1f hp=%.0f vel=%.2f%n", tick, u.x / 8f, u.y / 8f, Mathf.dst(u.x, u.y, b0.x, b0.y) / 8f, u.health, u.vel.len());
            }
        }
        System.out.printf("%s patch=%b controller=%s | margen minimo al alcance = %.1f casillas (negativo = dentro) | llego tick=%d | pos final (%.0f,%.0f) hp %.0f/%.0f%n",
            unitName, patch, ai.getClass().getSimpleName(), minMargin / 8f, arrived, u.x / 8f, u.y / 8f, u.health, startHp);
    }
}
