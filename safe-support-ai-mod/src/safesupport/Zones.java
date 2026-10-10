package safesupport;

import java.util.Arrays;

/**
 * Conjunto de zonas de peligro circulares (x, y, radio) y la geometria para esquivarlas.
 * No depende del juego, asi que se puede probar de forma aislada.
 */
public class Zones{
    /** El punto intermedio se coloca a radio * DETOUR + DETOUR_PAD del centro de la zona. */
    public static final float DETOUR = 1.12f, DETOUR_PAD = 8f;
    /** Margen extra al salir de una zona. */
    public static final float ESCAPE_PAD = 12f;

    private float[] data = new float[48];
    private int size; //valores usados: 3 por zona

    public void clear(){
        size = 0;
    }

    public int count(){
        return size / 3;
    }

    public void add(float x, float y, float radius){
        if(size + 3 > data.length) data = Arrays.copyOf(data, data.length * 2);
        data[size++] = x;
        data[size++] = y;
        data[size++] = radius;
    }

    /** @return true si el punto cae dentro de alguna zona (ampliada por margin). */
    public boolean dangerAt(float x, float y, float margin){
        for(int i = 0; i < size; i += 3){
            float dx = x - data[i], dy = y - data[i + 1], r = data[i + 2] + margin;
            if(dx * dx + dy * dy < r * r) return true;
        }
        return false;
    }

    /**
     * Si (px, py) esta dentro de alguna zona, escribe en out un punto fuera de ella y devuelve true.
     * Si esta dentro de varias, empuja la suma de todas.
     */
    public boolean escape(float px, float py, float[] out){
        float ox = 0f, oy = 0f;
        boolean any = false;

        for(int i = 0; i < size; i += 3){
            float cx = data[i], cy = data[i + 1], r = data[i + 2];
            float dx = px - cx, dy = py - cy;
            float d2 = dx * dx + dy * dy;
            if(d2 >= r * r) continue;

            any = true;
            float d = (float)Math.sqrt(d2);
            float nx, ny;
            if(d < 0.001f){
                nx = 1f;
                ny = 0f;
            }else{
                nx = dx / d;
                ny = dy / d;
            }

            float push = (r - d) + ESCAPE_PAD;
            ox += nx * push;
            oy += ny * push;
        }

        if(!any) return false;
        out[0] = px + ox;
        out[1] = py + oy;
        return true;
    }

    /**
     * Calcula el siguiente punto para ir de (px, py) a (tx, ty) sin atravesar ninguna zona.
     * - Si el destino esta dentro de una zona, el punto queda en el borde, del lado por el que se llega (espera).
     * - Si la recta cruza una zona, rodea por el lado mas cercano a la trayectoria.
     *
     * @return false si el camino directo es seguro (no hace falta desviarse); si no, escribe el punto en out.
     */
    public boolean waypoint(float px, float py, float tx, float ty, float[] out){
        boolean changed = false;

        //1) destino dentro de una zona: esperar en el borde
        for(int i = 0; i < size; i += 3){
            float cx = data[i], cy = data[i + 1], r = data[i + 2];
            float dxT = tx - cx, dyT = ty - cy;
            if(dxT * dxT + dyT * dyT >= r * r) continue;

            float dx = px - cx, dy = py - cy;
            float d = (float)Math.sqrt(dx * dx + dy * dy);
            if(d < r) continue; //ya estamos dentro: eso lo resuelve escape()

            tx = cx + dx / d * (r + DETOUR_PAD);
            ty = cy + dy / d * (r + DETOUR_PAD);
            changed = true;
        }

        //2) rodear las zonas que cruza la trayectoria (como mucho 3 desvios encadenados)
        for(int iter = 0; iter < 3; iter++){
            float sx = tx - px, sy = ty - py;
            float len2 = sx * sx + sy * sy;
            if(len2 < 1f) break;

            int hit = -1;
            float hitT = Float.MAX_VALUE, hqx = 0f, hqy = 0f;

            for(int i = 0; i < size; i += 3){
                float cx = data[i], cy = data[i + 1], r = data[i + 2];
                float ux = px - cx, uy = py - cy;
                if(ux * ux + uy * uy < r * r) continue; //ya dentro: lo resuelve escape()

                float t = ((cx - px) * sx + (cy - py) * sy) / len2;
                t = t < 0f ? 0f : (t > 1f ? 1f : t);
                float qx = px + sx * t, qy = py + sy * t;
                float ddx = qx - cx, ddy = qy - cy;

                if(ddx * ddx + ddy * ddy < r * r && t < hitT){
                    hit = i;
                    hitT = t;
                    hqx = qx;
                    hqy = qy;
                }
            }

            if(hit < 0) break;

            float cx = data[hit], cy = data[hit + 1], r = data[hit + 2];
            float nx = hqx - cx, ny = hqy - cy;
            float nl = (float)Math.sqrt(nx * nx + ny * ny);

            if(nl < 1f){
                //la recta pasa por el centro: lado fijo (perpendicular izquierda)
                float len = (float)Math.sqrt(len2);
                nx = -sy / len;
                ny = sx / len;
            }else{
                nx /= nl;
                ny /= nl;
            }

            tx = cx + nx * (r * DETOUR + DETOUR_PAD);
            ty = cy + ny * (r * DETOUR + DETOUR_PAD);
            changed = true;
        }

        if(!changed) return false;
        out[0] = tx;
        out[1] = ty;
        return true;
    }
}
