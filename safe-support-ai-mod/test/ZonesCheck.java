import safesupport.Zones;
public class ZonesCheck{
    static int fails;
    static void check(boolean c, String m){ if(!c){ fails++; System.out.println("FAIL: " + m); } else System.out.println("ok: " + m); }
    public static void main(String[] a){
        float[] o = new float[2];
        Zones z = new Zones();
        z.add(100, 0, 40);
        check(z.waypoint(0, 0, 200, 0, o), "waypoint exists crossing a zone");
        check(!z.dangerAt(o[0], o[1], 0), "waypoint is outside the zone");
        check(!z.waypoint(0, 100, 200, 100, o), "no waypoint when path is clear");
        // chain of zones
        Zones c = new Zones();
        for(int i = 0; i < 4; i++) c.add(100 + i * 60, 0, 40);
        float x = 0, y = 0; boolean safe = true;
        for(int step = 0; step < 50; step++){
            boolean w = c.waypoint(x, y, 400, 0, o);
            float tx = w ? o[0] : 400, ty = w ? o[1] : 0;
            // sample the segment
            for(int s = 1; s <= 20; s++){ float px = x + (tx - x) * s / 20f, py = y + (ty - y) * s / 20f; if(c.dangerAt(px, py, 0)) safe = false; }
            x = tx; y = ty; if(!w) break;
        }
        check(safe && x == 400, "chain of zones: path never enters a zone, reaches target");
        Zones d = new Zones(); d.add(0, 0, 50);
        check(d.escape(10, 0, o) && !d.dangerAt(o[0], o[1], 0), "escape from inside yields safe point");
        check(!d.escape(200, 0, o), "escape returns false when safe");
        System.exit(fails == 0 ? 0 : 1);
    }
}
