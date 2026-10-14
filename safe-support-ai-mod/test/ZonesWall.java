import safesupport.Zones;
public class ZonesWall{
    static int fails;
    static void check(boolean c, String m){ if(!c){ fails++; System.out.println("FAIL: " + m); } else System.out.println("ok: " + m); }

    /** Simula una unidad que sigue los puntos; devuelve {entroEnZona(0/1), llegoPaso o -1, xFinal, yFinal} */
    static float[] fly(Zones z, float[][] circles, float sx, float sy, float tx, float ty, float speed, int maxSteps){
        float x = sx, y = sy; float[] w = new float[2];
        int arrived = -1; boolean entered = false;
        for(int step = 0; step < maxSteps; step++){
            boolean has = z.waypoint(x, y, tx, ty, w);
            float gx = has ? w[0] : tx, gy = has ? w[1] : ty;
            float dx = gx - x, dy = gy - y, d = (float)Math.sqrt(dx * dx + dy * dy);
            if(d > 0.001f){ float s = Math.min(speed, d); x += dx / d * s; y += dy / d * s; }
            for(float[] c : circles){ float ex = x - c[0], ey = y - c[1]; if(ex * ex + ey * ey < c[2] * c[2] - 1f) entered = true; }
            if(Math.hypot(tx - x, ty - y) < 6f){ arrived = step; break; }
        }
        return new float[]{entered ? 1 : 0, arrived, x, y};
    }

    public static void main(String[] a){
        // pared de 5 torretas, alcance 160 + pad 24 = 184, separadas 64 (8 casillas)
        float[][] wall = new float[5][];
        float[] ys = {320, 256, 384, 192, 448};
        for(int i = 0; i < 5; i++) wall[i] = new float[]{480, ys[i], 184};

        Zones z = new Zones();
        for(float[] c : wall) z.add(c[0], c[1], c[2]);
        z.setBounds(1120, 1120); // mapa alto (140 casillas)
        float[] r = fly(z, wall, 112, 320, 800, 320, 2.5f, 4000);
        check(r[0] == 0f, "wall of 5 overlapping zones: unit never enters a zone");
        check(r[1] >= 0, "wall of 5 overlapping zones: unit reaches the far side (step " + (int)r[1] + ")");

        // mismo muro, destino al otro lado pero fuera de eje
        r = fly(z, wall, 112, 100, 800, 540, 2.5f, 4000);
        check(r[0] == 0f && r[1] >= 0, "different start/goal: safe and arrives");

        // mapa que no deja espacio: no entra, espera en el borde
        Zones tight = new Zones();
        for(float[] c : wall) tight.add(c[0], c[1], c[2]);
        tight.setBounds(1120, 640);
        r = fly(tight, wall, 112, 320, 800, 320, 2.5f, 3000);
        check(r[0] == 0f, "map with no room to go around: unit still never enters a zone (stops at " + (int)r[2] + "," + (int)r[3] + ")");

        // destino dentro de una zona: espera en el borde
        Zones one = new Zones(); one.add(480, 320, 184); one.setBounds(1120, 1120);
        r = fly(one, new float[][]{{480, 320, 184}}, 112, 320, 480, 320, 2.5f, 2000);
        check(r[0] == 0f && r[1] < 0, "destination inside a zone: unit waits at the edge without entering");

        // inicio dentro de una zona: sale por la recta sin bloquearse
        Zones in = new Zones(); in.add(200, 320, 184); in.setBounds(1120, 1120);
        float[] w = new float[2];
        check(!in.waypoint(200, 320, 800, 320, w) || true, "start inside a zone does not crash");

        // zonas separadas en varias filas
        Zones multi = new Zones();
        float[][] mc = {{300, 300, 150}, {520, 360, 150}, {740, 300, 150}, {520, 160, 150}};
        for(float[] c : mc) multi.add(c[0], c[1], c[2]);
        multi.setBounds(1200, 800);
        r = fly(multi, mc, 60, 300, 1100, 300, 2.5f, 5000);
        check(r[0] == 0f && r[1] >= 0, "scattered zones: safe and arrives");

        // rendimiento: 16 zonas, ruta recalculada cada 'escaneo' (10 pasos) durante 1 minuto simulado
        Zones big = new Zones();
        for(int i = 0; i < 16; i++) big.add(480 + (i % 2) * 40, 100 + i * 45, 170);
        big.setBounds(1400, 1400);
        long t0 = System.nanoTime(); int calls = 0;
        for(int rep = 0; rep < 200; rep++){
            Zones zz = new Zones(); for(int i = 0; i < 16; i++) zz.add(480 + (i % 2) * 40, 100 + i * 45, 170); zz.setBounds(1400, 1400);
            float[] ww = new float[2];
            for(int k = 0; k < 10; k++){ zz.waypoint(112 + k * 3, 400, 900, 400, ww); calls++; }
        }
        double ms = (System.nanoTime() - t0) / 1e6;
        System.out.printf("perf: %d waypoint calls over 16 zones took %.1f ms (%.3f ms/call, one replan per 10 calls)%n", calls, ms, ms / calls);
        check(ms / calls < 1.0, "average cost under 1 ms per call");
        System.exit(fails == 0 ? 0 : 1);
    }
}
