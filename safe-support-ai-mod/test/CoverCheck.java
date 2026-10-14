import safesupport.Cover;
public class CoverCheck{
    static int fails;
    static void check(boolean c, String m){ if(!c){ fails++; System.out.println("FAIL: " + m); } else System.out.println("ok: " + m); }
    static final float T = 8f;
    static float[] threat(float x, float y, float r, boolean blockable){ return new float[]{x, y, r, blockable ? 1 : 0}; }
    public static void main(String[] a){
        // muro vertical en x=10 casillas, de y=-3 a y=3
        Cover.Walls wall = (x, y) -> x == 10 && y >= -3 && y <= 3;
        float[] t = threat(0, 0, 400, true);
        check(Cover.covered(t, 1, wall, 14 * T, 0, 4, T), "unit directly behind wall is covered");
        check(!Cover.covered(t, 1, wall, 6 * T, 0, 4, T), "unit in front of wall is NOT covered");
        check(!Cover.covered(t, 1, wall, 14 * T, 8 * T, 4, T), "unit far past the wall end is NOT covered");
        check(!Cover.covered(t, 1, wall, 14 * T, 4.2f * T, 4, T), "unit just past the wall end (diagonal) is NOT covered");
        check(Cover.covered(t, 1, wall, 14 * T, 1.5f * T, 4, T), "unit slightly off-axis but behind wide wall is covered");
        // amenaza que atraviesa muros (laser / flak)
        check(!Cover.covered(threat(0, 0, 400, false), 1, wall, 14 * T, 0, 4, T), "unblockable threat: wall does not cover");
        // amenaza fuera de alcance no cuenta
        check(Cover.covered(threat(0, 0, 50, false), 1, wall, 14 * T, 0, 4, T), "threat that cannot reach the point is ignored");
        // dos amenazas: una tapada y otra no
        float[] two = {0, 0, 400, 1,  14 * T, 10 * T, 400, 1};
        check(!Cover.covered(two, 2, wall, 14 * T, 0, 4, T), "second threat from the side with no wall -> not covered");
        // hueco de una casilla en el muro
        Cover.Walls gap = (x, y) -> x == 10 && y >= -3 && y <= 3 && y != 0;
        check(!Cover.covered(t, 1, gap, 14 * T, 0, 4, T), "gap in the wall exposes the unit");
        // muro diagonal
        Cover.Walls diag = (x, y) -> x - y == 10 || x - y == 11;
        check(Cover.covered(threat(0, 0, 600, true), 1, diag, 12 * T, 0, 4, T) == Cover.blocked(diag, 0, 0, 12 * T, 0, 4, T), "covered() consistent with blocked()");
        // la casilla de la propia unidad / amenaza no cuenta como muro
        Cover.Walls selfTile = (x, y) -> x == 14 && y == 0;
        check(!Cover.covered(t, 1, selfTile, 14 * T, 0, 4, T), "wall under the unit itself does not count");
        // spot(): punto detras del muro, alejandose de la amenaza
        float[] out = new float[2];
        check(Cover.spot(10 * T, 0, T / 2, 0, 0, 1.5f * T, out) && Math.abs(out[0] - (10 * T + 2 * T)) < 0.01f && Math.abs(out[1]) < 0.01f, "spot is 2 tiles behind wall center, away from threat");
        check(!Cover.spot(5, 5, 4, 5, 5, 12, out), "spot rejects threat on top of wall");
        // spot generado queda cubierto
        Cover.spot(10 * T, 0, T / 2, 0, 0, 1.5f * T, out);
        check(Cover.covered(t, 1, wall, out[0], out[1], 4, T), "generated spot is covered");
        // spot diagonal
        Cover.Walls blk = (x, y) -> Math.abs(x - 8) <= 1 && Math.abs(y - 8) <= 1; // bloque 3x3
        Cover.spot(8 * T, 8 * T, 1.5f * T, 0, 0, 1.5f * T, out);
        check(Cover.covered(threat(0, 0, 900, true), 1, blk, out[0], out[1], 4, T), "spot behind a 3x3 block along the diagonal is covered");
        System.exit(fails == 0 ? 0 : 1);
    }
}
