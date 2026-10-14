import arc.*;
import arc.backend.headless.*;
import arc.struct.*;
import arc.util.*;
import arc.math.*;
import mindustry.*;
import mindustry.content.*;
import mindustry.core.*;
import mindustry.entities.units.*;
import mindustry.ai.*;
import mindustry.ai.types.*;
import mindustry.game.*;
import mindustry.gen.*;
import mindustry.net.*;
import mindustry.type.*;
import mindustry.world.*;
import mindustry.world.blocks.defense.turrets.*;
import safesupport.*;

import static mindustry.Vars.*;

public class Sim{
    static boolean ready;
    static Throwable err;

    public static void main(String[] args) throws Exception{
        Log.useColors = false;
        new HeadlessApplication(new ApplicationCore(){
            @Override public void setup(){
                headless = true;
                net = new Net(null);
                tree = new mindustry.core.FileTree();
                Vars.init();
                world = new World(){ public float getDarkness(int x, int y){ return 0; } };
                content.createBaseContent();
                add(logic = new Logic());
                add(netServer = new NetServer());
                content.init();
            }
            @Override public void init(){
                super.init();
                ready = true;
            }
        }, t -> err = t);

        long t0 = System.currentTimeMillis();
        while(!ready && err == null && System.currentTimeMillis() - t0 < 30000) Thread.sleep(50);
        if(err != null){ err.printStackTrace(); System.exit(2); }
        System.out.println("headless up");
        final float dl = Float.parseFloat(System.getProperty("delta", "1")); Time.setDeltaProvider(() -> dl);
        final String[] turrets = args.length > 1 ? args[1].split(",") : new String[]{"duo"};
        final String[] fa = args;
        final boolean[] done = {false};
        Core.app.post(() -> { try{ for(String tn : turrets) run(fa.length > 0 ? fa[0] : "mono", tn); }catch(Throwable t){ t.printStackTrace(); } done[0] = true; });
        while(!done[0]) Thread.sleep(20);
        System.exit(0);
    }

    static void run(String mode, String turretName) throws Exception{
        int W = 140, H = 60;
        logic.reset();
        world.beginMapLoad();
        Tiles tiles = world.resize(W, H);
        for(int x = 0; x < W; x++) for(int y = 0; y < H; y++) tiles.set(x, y, new Tile(x, y, Blocks.stone, Blocks.air, Blocks.air));
        //veta de cobre al otro lado de la torreta
        for(int x = 110; x < 114; x++) for(int y = 28; y < 32; y++) tiles.getn(x, y).setOverlay(Blocks.oreCopper);
        world.endMapLoad();

        state.rules.waves = false;
        state.rules.infiniteResources = true;

        tiles.getn(10, 30).setBlock(Blocks.coreShard, Team.sharded, 0);
        Tile tt = tiles.getn(60, 30);
        Block tb = mindustry.Vars.content.block(turretName);
        tt.setBlock(tb, Team.crux, 0);
        Turret.TurretBuild build = (Turret.TurretBuild)tt.build;
        if(build instanceof ItemTurret.ItemTurretBuild ib){
            for(Item it : ((ItemTurret)tb).ammoTypes.keys()){ for(int i = 0; i < 20; i++) ib.handleItem(ib, it); break; }
        }
        if(System.getProperty("cluster") != null){
            int[][] extra = {{60,24},{60,36},{66,30},{54,30},{60,18},{60,42}};
            for(int[] e : extra){
                Tile et = tiles.getn(e[0], e[1]);
                et.setBlock(tb, Team.crux, 0);
                if(et.build instanceof ItemTurret.ItemTurretBuild ib2){ for(Item it : ((ItemTurret)tb).ammoTypes.keys()){ for(int i = 0; i < 20; i++) ib2.handleItem(ib2, it); break; } }
            }
        }
        Turret tr = (Turret)tb;
        System.out.println("== " + mode + " vs " + turretName + " targetAir=" + tr.targetAir + " range=" + build.range() / 8f + " tiles");

        if(mode.equals("mega")){
            //bloque danado al otro lado de la torreta
            Tile wt = tiles.getn(100, 30);
            wt.setBlock(Blocks.copperWallLarge, Team.sharded, 0);
            wt.build.health = wt.build.maxHealth * 0.4f;
            indexer.notifyHealthChanged(wt.build);
        }else if(mode.equals("poly")){
            //plan de reconstruccion al otro lado de la torreta
            state.teams.get(Team.sharded).plans.addLast(new Teams.BlockPlan(100, 30, (short)0, Blocks.copperWall, null));
        }
        state.set(GameState.State.playing);
        UnitType type = mode.equals("mega") ? UnitTypes.mega : mode.equals("poly") ? UnitTypes.poly : UnitTypes.mono;
        Unit u = type.spawn(Team.sharded, 14 * 8, 30 * 8);
        AIController ai;
        if(System.getProperty("cmd") != null){
            //igual que el juego: parche del mod + CommandAI con una orden
            var pm = SafeSupportMod.class.getDeclaredMethod("patch"); pm.setAccessible(true); pm.invoke(null);
            CommandAI cai = new CommandAI();
            u.controller(cai);
            cai.command(mode.equals("mega") ? UnitCommand.repairCommand : mode.equals("poly") ? UnitCommand.rebuildCommand : UnitCommand.mineCommand);
            cai.updateUnit(); //crea el controlador de la orden
            var cf = CommandAI.class.getDeclaredField("commandController"); cf.setAccessible(true);
            ai = (AIController)cf.get(cai);
            System.out.println("command controller = " + ai.getClass().getName());
        }else if(System.getProperty("plain") != null){
            ai = mode.equals("mega") ? new RepairAI() : mode.equals("poly") ? new BuilderAI() : new MinerAI();
            u.controller(ai);
        }else{
            ai = mode.equals("mega") ? new SafeRepairAI() : mode.equals("poly") ? new SafeBuilderAI() : new SafeMinerAI();
            u.controller(ai);
        }
        System.out.println("spawned " + type.name + " core=" + u.closestCore() + " ai=" + ai.getClass().getSimpleName());

        Seq<Building> tl = new Seq<>(); for(Building b : indexer.getEnemy(Team.sharded, mindustry.world.meta.BlockFlag.turret)) tl.add(b);
        float minMargin = Float.MAX_VALUE; int marginTick = -1; float mx = 0, my = 0; int firstHit = -1; float hitX = 0, hitY = 0;
        float minDst = Float.MAX_VALUE, tx = 60 * 8 + 4, ty = 30 * 8 + 4;
        float startHp = u.health;
        for(int tick = 0; tick < 6000; tick++){
            logic.update();
            if(!u.isValid()){ System.out.println("UNIT DIED at tick " + tick); break; }
            for(Building b : tl){ float mg = Mathf.dst(u.x, u.y, b.x, b.y) - ((Turret.TurretBuild)b).range(); if(mg < minMargin){ minMargin = mg; marginTick = tick; mx = u.x / 8f; my = u.y / 8f; } }
            if(firstHit < 0 && u.health < startHp - 0.5f){ firstHit = tick; hitX = u.x / 8f; hitY = u.y / 8f; }
            float d = Mathf.dst(u.x, u.y, tx, ty);
            minDst = Math.min(minDst, d);
            if(tick == 200) System.out.println("  at t=200 hasAmmo=" + build.hasAmmo());
            if(System.getProperty("trace") != null && tick % 20 == 0 && tick < 900) System.out.printf("  t=%d pos=(%.1f,%.1f) dstTurret=%.1f hp=%.0f vel=%.2f target=%s%n", tick, u.x / 8f, u.y / 8f, d / 8f, u.health, u.vel.len(), ai instanceof SafeRepairAI ? "-" : "-");
        }
        System.out.printf("  minMargin over all turrets = %.1f tiles at tick %d pos(%.0f,%.0f); first damage tick %d at (%.0f,%.0f)%n", minMargin / 8f, marginTick, mx, my, firstHit, hitX, hitY);
        System.out.printf("  RESULT minDst=%.1f tiles (range %.1f) hp %.0f/%.0f %s%n", minDst / 8f, build.range() / 8f, u.health, startHp, minDst < build.range() ? "<<< ENTRO EN EL ALCANCE" : "ok");

        //estado interno del guard
        try{
            var f = AIController.class; // no-op
            var gf = ai.getClass().getDeclaredField("guard"); gf.setAccessible(true);
            Object g = gf.get(ai);
            var tf = SafetyGuard.class.getDeclaredField("threats"); tf.setAccessible(true);
            System.out.println("  guard threats = " + ((arc.struct.FloatSeq)tf.get(g)).size / 4);
        }catch(Throwable t){ System.out.println("guard introspection: " + t); }
    }
}
