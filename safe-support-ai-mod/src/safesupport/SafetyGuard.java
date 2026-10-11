package safesupport;

import arc.*;
import arc.graphics.*;
import arc.graphics.g2d.*;
import arc.math.*;
import arc.math.geom.*;
import arc.struct.*;
import arc.util.*;
import mindustry.entities.*;
import mindustry.entities.bullet.*;
import mindustry.game.*;
import mindustry.gen.*;
import mindustry.graphics.*;
import mindustry.type.*;
import mindustry.core.*;
import mindustry.world.blocks.ConstructBlock.*;
import mindustry.world.blocks.defense.*;
import mindustry.world.blocks.defense.turrets.*;
import mindustry.world.meta.*;

import static mindustry.Vars.*;

/**
 * Detecta enemigos y torretas enemigas cerca de una unidad y los convierte en zonas de peligro.
 * Tambien decide si un muro propio protege a la unidad de esas amenazas.
 * Cada unidad de soporte tiene su propia instancia.
 */
public class SafetyGuard{
    /** Radio de busqueda de amenazas alrededor de la unidad. */
    public static float scanRadius = 80f * tilesize;
    /** Margen que se suma al alcance de una unidad enemiga. */
    public static float unitPad = 4f * tilesize;
    /** Margen que se suma al alcance de una torreta enemiga. */
    public static float turretPad = 3f * tilesize;
    /** Radio minimo de peligro de una unidad enemiga (por si su alcance es muy corto). */
    public static float minUnitDanger = 6f * tilesize;
    /** Cada cuantos ticks se vuelve a escanear. */
    public static float scanInterval = 10f;

    /** Hasta donde busca muros para cubrirse. */
    public static float coverSearch = 10f * tilesize;
    /** Distancia entre el borde del muro y la unidad (las balas que explotan al chocar hacen dano en area). */
    public static float coverGap = 1.5f * tilesize;
    /** Separacion de las lineas de tiro que se comprueban a cada lado de la unidad. */
    public static float coverSpread = 4f;
    /** Cada cuantos ticks se vuelve a buscar un muro. */
    public static float coverInterval = 30f;
    /** Cuantos puntos candidatos se comprueban como maximo en cada busqueda. */
    public static int coverChecks = 16;
    /** Cuanto mas lejos que la salida de la zona se acepta un muro. */
    public static float coverDetour = 5f * tilesize;

    /** Hasta donde busca el mono una veta alternativa, medido desde el nucleo. */
    public static float oreSearch = 40f * tilesize;
    /** Muestra las zonas y la cobertura sobre el mapa. */
    public static boolean debug;

    private static final int maxCandidates = 256;
    private static final int maxDebugGuards = 8;
    private static final Seq<SafetyGuard> tracked = new Seq<>();
    private static float lastSettingsRead = -1000f;

    private final Zones zones = new Zones();
    private final FloatSeq threats = new FloatSeq();
    private final float[] tmp = new float[2];
    private final Vec2 escape = new Vec2(), waypoint = new Vec2(), cover = new Vec2();
    private float lastScan = -1000f;

    //cobertura
    private final float[] candidates = new float[maxCandidates * 3];
    private Team coverTeam = Team.derelict;
    private final Cover.Walls walls = (x, y) -> isWall(world.build(x, y), coverTeam);
    private boolean hasCover, coveredCache;
    private float lastCoverSearch = -1000f, coveredStamp = -1f;

    //cobertura cerca de un bloque danado (para curarlo desde detras de un muro)
    private final Vec2 nearCover = new Vec2();
    private boolean hasNear;
    private float lastNearSearch = -1000f, nearAx, nearAy;

    //diagnostico
    private Unit owner;
    private boolean isTracked;

    /** Reescanea las amenazas si ya toca. Es barato llamarlo en cada frame. */
    public void update(Unit unit){
        float now = Time.time;
        if(now >= lastScan && now - lastScan < scanInterval) return;
        lastScan = now;

        readSettings();
        owner = unit;
        if(debug && !isTracked){
            isTracked = true;
            tracked.add(this);
        }

        zones.clear();
        threats.clear();
        final boolean air = unit.isFlying();

        //unidades enemigas que pueden dispararle a esta unidad
        Units.nearbyEnemies(unit.team, unit.x, unit.y, scanRadius, e -> {
            if(!e.isValid() || !e.type.canAttack) return;
            if(air ? !e.type.targetAir : !e.type.targetGround) return;

            float range = e.type.maxRange > 0f ? e.type.maxRange : Math.max(e.type.range, 0f);
            float radius = Math.max(minUnitDanger, range) + unitPad;
            zones.add(e.x, e.y, radius);
            addThreat(e.x, e.y, radius, unitBlockable(e.type, air));
        });

        //torretas enemigas con municion que pueden alcanzarla
        Seq<Building> turrets = indexer.getEnemy(unit.team, BlockFlag.turret);
        float limit = scanRadius + 20f * tilesize;
        float limit2 = limit * limit;

        for(int i = 0; i < turrets.size; i++){
            Building b = turrets.get(i);
            if(b == null || !b.isValid()) continue;

            float dx = b.x - unit.x, dy = b.y - unit.y;
            if(dx * dx + dy * dy > limit2) continue;

            if(!(b.block instanceof Turret t)) continue;
            if(!(b instanceof Turret.TurretBuild tb)) continue;
            if(air ? !t.targetAir : !t.targetGround) continue;
            if(!tb.hasAmmo()) continue;

            float radius = tb.range() + turretPad;
            zones.add(b.x, b.y, radius);
            addThreat(b.x, b.y, radius, bulletBlockable(tb.peekAmmo()));
        }

        coveredStamp = -1f;
    }

    private void addThreat(float x, float y, float radius, boolean blockable){
        threats.add(x);
        threats.add(y);
        threats.add(radius);
        threats.add(blockable ? 1f : 0f);
    }

    /** @return true si hay alguna amenaza detectada cerca. */
    public boolean hasThreats(){
        return zones.count() > 0;
    }

    /** @return true si el punto (en coordenadas de mundo) cae dentro de una zona de peligro. */
    public boolean dangerAt(float x, float y, float margin){
        return zones.dangerAt(x, y, margin);
    }

    /** @return un punto seguro al que huir si la unidad esta dentro de una zona; null si esta a salvo. */
    public @Nullable Vec2 escapePoint(Unit unit){
        if(!zones.escape(unit.x, unit.y, tmp)) return null;
        return escape.set(tmp[0], tmp[1]);
    }

    /** @return un punto intermedio para rodear las zonas hacia el destino; null si el camino directo es seguro. */
    public @Nullable Vec2 waypoint(Unit unit, Position target){
        if(!zones.waypoint(unit.x, unit.y, target.getX(), target.getY(), tmp)) return null;
        return waypoint.set(tmp[0], tmp[1]);
    }

    /** @return true si un muro propio frena a todas las amenazas que alcanzan a la unidad en su posicion actual. */
    public boolean covered(Unit unit){
        if(coveredStamp != lastScan){
            coverTeam = unit.team;
            coveredCache = Cover.covered(threats.items, threats.size / Cover.STRIDE, walls, unit.x, unit.y, coverSpread, tilesize);
            coveredStamp = lastScan;
        }
        return coveredCache;
    }

    /** @return un punto protegido por un muro cercano; null si no hay ninguno. Se recalcula cada coverInterval ticks. */
    public @Nullable Vec2 coverPoint(Unit unit){
        float now = Time.time;
        if(now >= lastCoverSearch && now - lastCoverSearch < coverInterval) return hasCover ? cover : null;

        lastCoverSearch = now;
        hasCover = searchCover(unit, unit.x, unit.y, -1f, cover);
        return hasCover ? cover : null;
    }

    /**
     * @return un punto protegido por un muro a menos de range del punto (ax, ay), por ejemplo para curar desde ahi un
     * bloque danado que esta en una zona de peligro; null si no hay ninguno. Se recalcula cada coverInterval ticks
     * mientras el punto no cambie.
     */
    public @Nullable Vec2 coverNear(Unit unit, float ax, float ay, float range){
        float now = Time.time;
        boolean same = Mathf.dst(ax, ay, nearAx, nearAy) < tilesize;
        if(same && now >= lastNearSearch && now - lastNearSearch < coverInterval) return hasNear ? nearCover : null;

        lastNearSearch = now;
        nearAx = ax;
        nearAy = ay;
        hasNear = searchCover(unit, ax, ay, range, nearCover);
        return hasNear ? nearCover : null;
    }

    /**
     * Busca muros alrededor de (ax, ay) y escribe en out el punto cubierto mas cercano a la unidad.
     * Con anchorRange > 0 solo acepta puntos a menos de esa distancia de (ax, ay).
     */
    private boolean searchCover(Unit unit, float ax, float ay, float anchorRange, Vec2 out){
        coverTeam = unit.team;

        boolean anchored = anchorRange > 0f;
        int cx = World.toTile(ax), cy = World.toTile(ay), r = (int)((anchored ? anchorRange : coverSearch) / tilesize) + (anchored ? 1 : 0);
        float maxX = world.unitWidth(), maxY = world.unitHeight();
        int n = 0;

        //un candidato por muro: justo detras de el, visto desde la amenaza mas cercana a ese muro
        for(int x = cx - r; x <= cx + r && n < maxCandidates; x++){
            for(int y = cy - r; y <= cy + r && n < maxCandidates; y++){
                Building b = world.build(x, y);
                if(!isWall(b, coverTeam) || b.tile.x != x || b.tile.y != y) continue;

                int near = nearestThreat(b.x, b.y);
                if(near < 0) continue;

                int k = near * Cover.STRIDE;
                if(!Cover.spot(b.x, b.y, b.block.size * tilesize / 2f, threats.items[k], threats.items[k + 1], coverGap, tmp)) continue;
                if(tmp[0] < 0f || tmp[1] < 0f || tmp[0] > maxX || tmp[1] > maxY) continue;
                if(anchored && Mathf.dst2(tmp[0], tmp[1], ax, ay) > anchorRange * anchorRange) continue;

                candidates[n * 3] = tmp[0];
                candidates[n * 3 + 1] = tmp[1];
                candidates[n * 3 + 2] = Mathf.dst2(tmp[0], tmp[1], unit.x, unit.y);
                n++;
            }
        }

        //el mas cercano a la unidad que de verdad quede cubierto
        for(int checks = 0; checks < coverChecks; checks++){
            int best = -1;
            for(int i = 0; i < n; i++){
                if(candidates[i * 3 + 2] >= 0f && (best < 0 || candidates[i * 3 + 2] < candidates[best * 3 + 2])) best = i;
            }
            if(best < 0) break;

            float px = candidates[best * 3], py = candidates[best * 3 + 1];
            candidates[best * 3 + 2] = -1f;

            if(Cover.covered(threats.items, threats.size / Cover.STRIDE, walls, px, py, coverSpread, tilesize)){
                out.set(px, py);
                return true;
            }
        }

        return false;
    }

    /** El bloque propio danado mas cercano a la unidad dentro de range (aunque este en una zona de peligro). */
    static @Nullable Building damagedNear(Unit unit, float range){
        var list = indexer.getDamaged(unit.team);
        Building best = null;
        float bestDst = range * range;

        for(int i = 0; i < list.size; i++){
            Building b = list.get(i);
            if(b == null || !b.isValid() || b instanceof ConstructBuild) continue;

            float dst = b.dst2(unit.x, unit.y);
            if(dst >= bestDst) continue;

            best = b;
            bestDst = dst;
        }

        return best;
    }

    /** @return indice de la amenaza mas cercana al punto que lo alcanza, o -1. */
    private int nearestThreat(float x, float y){
        int best = -1;
        float bestDst = Float.MAX_VALUE;
        for(int i = 0; i < threats.size / Cover.STRIDE; i++){
            int k = i * Cover.STRIDE;
            float dst = Mathf.dst2(x, y, threats.items[k], threats.items[k + 1]);
            float r = threats.items[k + 2];
            if(dst <= r * r && dst < bestDst){
                best = i;
                bestDst = dst;
            }
        }
        return best;
    }

    /** Lee los ajustes del juego (como mucho una vez por segundo para todas las unidades). */
    static void readSettings(){
        float now = Time.time;
        if(now >= lastSettingsRead && now - lastSettingsRead < 60f) return;
        lastSettingsRead = now;

        var s = Core.settings;
        scanRadius = s.getInt("ss-scan", 80) * tilesize;
        unitPad = s.getInt("ss-unitpad", 4) * tilesize;
        turretPad = s.getInt("ss-turretpad", 3) * tilesize;
        minUnitDanger = s.getInt("ss-minunit", 6) * tilesize;
        coverSearch = s.getInt("ss-coversearch", 10) * tilesize;
        coverGap = s.getInt("ss-covergap", 15) / 10f * tilesize;
        oreSearch = s.getInt("ss-oresearch", 40) * tilesize;
        debug = s.getBool("ss-debug", false);
    }

    /**
     * Dibuja las zonas de peligro de las unidades seleccionadas (o de las primeras si no hay seleccion):
     * naranja = la amenaza la frena un muro, rojo = atraviesa muros; anillo verde = unidad cubierta,
     * anillo rojo = expuesta; punto y linea verdes = muro elegido para cubrirse.
     */
    static void drawDebug(){
        if(!debug || tracked.isEmpty()) return;

        for(int i = tracked.size - 1; i >= 0; i--){
            SafetyGuard g = tracked.get(i);
            //sin escaneos recientes: la unidad cambio de IA o murio
            if(g.owner == null || !g.owner.isValid() || Time.time - g.lastScan > 120f){
                g.isTracked = false;
                tracked.remove(i);
            }
        }

        Seq<Unit> selected = control != null && control.input != null ? control.input.selectedUnits : null;
        boolean filter = selected != null && selected.size > 0;

        Draw.z(Layer.overlayUI);
        int drawn = 0;
        for(int i = 0; i < tracked.size && drawn < maxDebugGuards; i++){
            SafetyGuard g = tracked.get(i);
            if(filter && !selected.contains(g.owner)) continue;
            g.drawZones();
            drawn++;
        }
        Draw.reset();
    }

    private void drawZones(){
        Lines.stroke(1.5f);
        for(int i = 0; i < threats.size / Cover.STRIDE; i++){
            int k = i * Cover.STRIDE;
            Draw.color(threats.items[k + 3] > 0f ? Color.orange : Color.scarlet, 0.9f);
            Lines.circle(threats.items[k], threats.items[k + 1], threats.items[k + 2]);
            Fill.circle(threats.items[k], threats.items[k + 1], 3f);
        }

        if(zones.dangerAt(owner.x, owner.y, 0f)){
            Draw.color(covered(owner) ? Color.green : Color.red, 0.9f);
            Lines.stroke(2f);
            Lines.circle(owner.x, owner.y, owner.hitSize / 2f + 4f);

            if(hasCover){
                Draw.color(Color.green, 0.9f);
                Lines.stroke(1f);
                Lines.line(owner.x, owner.y, cover.x, cover.y);
                Fill.circle(cover.x, cover.y, 3f);
            }
        }
    }

    /** Muro propio que frena balas: de la categoria muros, sin puertas ni bloques que las balas sobrevuelan. */
    static boolean isWall(@Nullable Building b, Team team){
        return b != null && b.team == team && b.isValid()
            && b.block.group == BlockGroup.walls && !b.block.underBullets && !(b.block instanceof Door);
    }

    /** Una bala la frena un muro si choca con edificios; los laseres y la metralla antiaerea pasan de largo. */
    static boolean bulletBlockable(@Nullable BulletType b){
        return b != null && b.collides && b.collidesTiles && b.collidesGround && !b.pierceBuilding;
    }

    /** @return true si todas las armas de la unidad que pueden herir a la unidad objetivo disparan balas bloqueables. */
    static boolean unitBlockable(UnitType type, boolean targetIsAir){
        boolean any = false;
        for(Weapon w : type.weapons){
            BulletType b = w.bullet;
            if(b == null || (targetIsAir ? !b.collidesAir : !b.collidesGround)) continue;
            if(!bulletBlockable(b)) return false;
            any = true;
        }
        return any;
    }
}
