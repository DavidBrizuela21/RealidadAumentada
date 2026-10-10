package safesupport;

/**
 * Geometria de cobertura detras de muros: decide si una posicion esta protegida de todas las
 * amenazas que la alcanzan. No depende del juego, asi que se puede probar de forma aislada.
 *
 * Una amenaza se guarda como 4 valores seguidos: x, y, radio de peligro y 1 si sus balas las
 * frena un muro (0 si las atraviesa, como los laseres o la metralla antiaerea).
 */
public class Cover{
    /** Valores por amenaza en el arreglo de amenazas. */
    public static final int STRIDE = 4;

    /** Dice si una casilla (en coordenadas de casilla) tiene un muro que frena las balas. */
    public interface Walls{
        boolean wall(int tx, int ty);
    }

    /**
     * @return true si todas las amenazas que alcanzan (px, py) son bloqueables y tienen un muro en medio.
     * Sin amenazas que la alcancen tambien devuelve true: el punto es seguro.
     * @param spread separacion de las lineas paralelas que representan el ancho de la unidad
     * @param tile tamano de una casilla en unidades de mundo
     */
    public static boolean covered(float[] threats, int count, Walls walls, float px, float py, float spread, float tile){
        for(int i = 0; i < count; i++){
            int k = i * STRIDE;
            float tx = threats[k], ty = threats[k + 1], r = threats[k + 2];
            float dx = px - tx, dy = py - ty;
            if(dx * dx + dy * dy > r * r) continue; //no la alcanza

            if(threats[k + 3] <= 0f) return false; //sus balas atraviesan muros
            if(!blocked(walls, tx, ty, px, py, spread, tile)) return false;
        }
        return true;
    }

    /** Una linea central y dos paralelas a cada lado: las tres tienen que topar con un muro. */
    public static boolean blocked(Walls walls, float x0, float y0, float x1, float y1, float spread, float tile){
        float dx = x1 - x0, dy = y1 - y0;
        float len = (float)Math.sqrt(dx * dx + dy * dy);
        if(len < 1f) return false;

        float nx = -dy / len * spread, ny = dx / len * spread;
        return lineHitsWall(walls, x0, y0, x1, y1, tile)
            && lineHitsWall(walls, x0 + nx, y0 + ny, x1 + nx, y1 + ny, tile)
            && lineHitsWall(walls, x0 - nx, y0 - ny, x1 - nx, y1 - ny, tile);
    }

    /** Recorre las casillas entre los dos puntos (sin contar la primera ni la ultima) buscando un muro. */
    public static boolean lineHitsWall(Walls walls, float x0, float y0, float x1, float y1, float tile){
        int sx0 = Math.round(x0 / tile), sy0 = Math.round(y0 / tile);
        int xe = Math.round(x1 / tile), ye = Math.round(y1 / tile);

        int x = sx0, y = sy0;
        int dx = Math.abs(xe - x), dy = Math.abs(ye - y);
        int sx = x < xe ? 1 : -1, sy = y < ye ? 1 : -1;
        int err = dx - dy;

        while(true){
            boolean endpoint = (x == sx0 && y == sy0) || (x == xe && y == ye);
            if(!endpoint && walls.wall(x, y)) return true;
            if(x == xe && y == ye) return false;

            int e2 = 2 * err;
            if(e2 > -dy){ err -= dy; x += sx; }
            if(e2 < dx){ err += dx; y += sy; }
        }
    }

    /**
     * Punto detras de un muro: desde el centro del muro (wx, wy), alejandose de la amenaza (tx, ty),
     * a media anchura del muro mas gap. Escribe el punto en out y devuelve true; false si coinciden.
     */
    public static boolean spot(float wx, float wy, float halfSize, float tx, float ty, float gap, float[] out){
        float dx = wx - tx, dy = wy - ty;
        float len = (float)Math.sqrt(dx * dx + dy * dy);
        if(len < 1f) return false;

        float d = halfSize + gap;
        out[0] = wx + dx / len * d;
        out[1] = wy + dy / len * d;
        return true;
    }
}
