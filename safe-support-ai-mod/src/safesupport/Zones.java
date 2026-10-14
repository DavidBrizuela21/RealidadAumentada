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
    /** Distancia al borde de una zona a la que se espera (una unidad rapida se pasa al frenar). */
    public static final float HOLD_PAD = 24f;

    private float[] data = new float[48];
    private int size; //valores usados: 3 por zona

    //limites del mapa (0 = sin limites) y ruta guardada
    private float boundsW, boundsH;
    private int version;
    private int routeVersion = -1;
    private float routeTx, routeTy;
    private float[] route = new float[0];
    private int routeLen, routeIdx; //puntos de la ruta (x, y por punto) y siguiente punto a seguir

    //memoria de trabajo del planificador
    private boolean[] skip = new boolean[0], inc = new boolean[0];
    private float[] nx = new float[0], ny = new float[0], g = new float[0];
    private int[] prev = new int[0], queue = new int[0];
    private boolean[] closed = new boolean[0];

    public void clear(){
        size = 0;
        version++;
    }

    /** Limites del mapa en unidades de mundo: los puntos de desvio fuera de ellos se descartan. 0 = sin limites. */
    public void setBounds(float width, float height){
        boundsW = width;
        boundsH = height;
    }

    public int count(){
        return size / 3;
    }

    public void add(float x, float y, float radius){
        if(size + 3 > data.length) data = Arrays.copyOf(data, data.length * 2);
        data[size++] = x;
        data[size++] = y;
        data[size++] = radius;
        version++;
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

    /** Candidatos por zona en la planificacion de rutas y maximo de zonas que se planifican a la vez. */
    static final int RING = 8, MAX_CIRCLES = 16;
    /** Distancia a un punto de la ruta a la que se da por alcanzado. */
    static final float REACH = 12f;

    /**
     * Calcula el siguiente punto para ir de (px, py) a (tx, ty) sin atravesar ninguna zona, rodeando incluso
     * varias zonas solapadas (por ejemplo una pared de torretas).
     * - Si el destino esta dentro de una zona, el punto queda en el borde, del lado por el que se llega (espera).
     * - Si la recta cruza zonas, planifica una ruta por puntos alrededor de las zonas y devuelve el siguiente.
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

            tx = cx + dx / d * (r + HOLD_PAD);
            ty = cy + dy / d * (r + HOLD_PAD);
            changed = true;
        }

        int n = size / 3;
        if(skip.length < n) skip = new boolean[n * 2];
        for(int i = 0; i < n; i++){
            float dx = px - data[i * 3], dy = py - data[i * 3 + 1];
            skip[i] = dx * dx + dy * dy < data[i * 3 + 2] * data[i * 3 + 2]; //zonas en las que ya estamos: las resuelve escape()
        }

        //2) recta libre: nada que rodear
        if(segmentFree(px, py, tx, ty)){
            routeVersion = -1;
            if(!changed) return false;
            out[0] = tx;
            out[1] = ty;
            return true;
        }

        //3) ruta guardada, o planificar una nueva
        if(!(routeVersion == version && Math.abs(routeTx - tx) < 1f && Math.abs(routeTy - ty) < 1f) || !followRoute(px, py)){
            if(!plan(px, py, tx, ty)){
                routeVersion = -1;
                //sin ruta (por ejemplo el mapa no deja espacio): rodeo simple si es seguro, si no espera en el borde
                if(legacyWaypoint(px, py, tx, ty, out) && segmentFree(px, py, out[0], out[1])) return true;
                holdAtEdge(px, py, tx, ty, out);
                return true;
            }
            followRoute(px, py);
        }

        out[0] = route[routeIdx * 2];
        out[1] = route[routeIdx * 2 + 1];
        return true;
    }

    /** Sin camino seguro: el punto queda justo antes de donde la recta entra en la primera zona (en el borde de la union de zonas). */
    private void holdAtEdge(float px, float py, float tx, float ty, float[] out){
        float sx = tx - px, sy = ty - py;
        float len = (float)Math.sqrt(sx * sx + sy * sy);
        out[0] = px;
        out[1] = py;
        if(len < 0.001f) return;

        float ux = sx / len, uy = sy / len;
        float entry = Float.MAX_VALUE; //distancia a lo largo de la recta hasta la primera zona

        for(int i = 0; i < size / 3; i++){
            if(skip[i]) continue;
            float cx = data[i * 3], cy = data[i * 3 + 1], r = data[i * 3 + 2];
            float ox = px - cx, oy = py - cy;
            float b = ox * ux + oy * uy;
            float c = ox * ox + oy * oy - r * r;
            float disc = b * b - c;
            if(disc < 0f) continue;
            float s0 = -b - (float)Math.sqrt(disc);
            if(s0 < 0f || s0 > len) continue;
            if(s0 < entry) entry = s0;
        }

        if(entry == Float.MAX_VALUE) return;
        float d = Math.max(0f, entry - HOLD_PAD);
        out[0] = px + ux * d;
        out[1] = py + uy * d;
    }

    /** Avanza al siguiente punto de la ruta guardada. @return false si el punto actual ya no se puede alcanzar en linea recta. */
    private boolean followRoute(float px, float py){
        while(routeIdx < routeLen - 1){
            float dx = route[routeIdx * 2] - px, dy = route[routeIdx * 2 + 1] - py;
            boolean reached = dx * dx + dy * dy < REACH * REACH;
            //si ya se ve el punto siguiente en linea recta, no hace falta pasar por este
            boolean nextVisible = segmentFree(px, py, route[routeIdx * 2 + 2], route[routeIdx * 2 + 3], 1.05f, 6f);
            if(!reached && !nextVisible) break;
            routeIdx++;
        }
        return segmentFree(px, py, route[routeIdx * 2], route[routeIdx * 2 + 1]);
    }

    /** @return true si el segmento no entra en ninguna zona (ignorando aquellas en las que esta el inicio). */
    private boolean segmentFree(float ax, float ay, float bx, float by){
        return segmentFree(ax, ay, bx, by, 1f, 0f);
    }

    /** Igual, pero exige pasar a al menos radio * scale + add del centro de cada zona. */
    private boolean segmentFree(float ax, float ay, float bx, float by, float scale, float add){
        float sx = bx - ax, sy = by - ay;
        float len2 = sx * sx + sy * sy;
        int n = size / 3;

        for(int i = 0; i < n; i++){
            if(skip[i]) continue;
            float cx = data[i * 3], cy = data[i * 3 + 1], r = data[i * 3 + 2] * scale + add;

            float t = len2 < 0.0001f ? 0f : ((cx - ax) * sx + (cy - ay) * sy) / len2;
            t = t < 0f ? 0f : (t > 1f ? 1f : t);
            float dx = ax + sx * t - cx, dy = ay + sy * t - cy;
            if(dx * dx + dy * dy < r * r - 1f) return false;
        }
        return true;
    }

    /**
     * Busca el camino mas corto de (px, py) a (tx, ty) pasando por puntos situados alrededor de las zonas que estorban
     * (A* sobre un grafo de visibilidad). Guarda la ruta. @return false si no hay camino.
     */
    private boolean plan(float px, float py, float tx, float ty){
        int n = size / 3;
        if(inc.length < n){
            inc = new boolean[n * 2];
            queue = new int[n * 2];
        }
        java.util.Arrays.fill(inc, 0, n, false);

        //zonas que estorban: las que cruza la recta y, de forma encadenada, las que se solapan con ellas
        int count = 0, head = 0;
        for(int i = 0; i < n && count < MAX_CIRCLES; i++){
            if(skip[i]) continue;
            float cx = data[i * 3], cy = data[i * 3 + 1], r = data[i * 3 + 2];
            float sx = tx - px, sy = ty - py, len2 = sx * sx + sy * sy;
            float t = len2 < 0.0001f ? 0f : ((cx - px) * sx + (cy - py) * sy) / len2;
            t = t < 0f ? 0f : (t > 1f ? 1f : t);
            float dx = px + sx * t - cx, dy = py + sy * t - cy;
            if(dx * dx + dy * dy < r * r){
                inc[i] = true;
                queue[count++] = i;
            }
        }
        while(head < count && count < MAX_CIRCLES){
            int a = queue[head++];
            for(int j = 0; j < n && count < MAX_CIRCLES; j++){
                if(inc[j] || skip[j]) continue;
                float dx = data[a * 3] - data[j * 3], dy = data[a * 3 + 1] - data[j * 3 + 1];
                float rr = data[a * 3 + 2] + data[j * 3 + 2];
                if(dx * dx + dy * dy < rr * rr){
                    inc[j] = true;
                    queue[count++] = j;
                }
            }
        }

        //nodos: 0 = inicio, 1 = destino, luego RING puntos alrededor de cada zona elegida
        int max = 2 + count * RING;
        if(nx.length < max){
            nx = new float[max * 2];
            ny = new float[max * 2];
            g = new float[max * 2];
            prev = new int[max * 2];
            closed = new boolean[max * 2];
        }
        nx[0] = px; ny[0] = py;
        nx[1] = tx; ny[1] = ty;
        int v = 2;
        for(int q = 0; q < count; q++){
            int i = queue[q];
            float cx = data[i * 3], cy = data[i * 3 + 1], R = data[i * 3 + 2] * DETOUR + DETOUR_PAD;
            for(int k = 0; k < RING; k++){
                float a = k * (float)(Math.PI * 2 / RING);
                float x = cx + (float)Math.cos(a) * R, y = cy + (float)Math.sin(a) * R;
                if(boundsW > 0f && (x < 4f || y < 4f || x > boundsW - 4f || y > boundsH - 4f)) continue;
                if(insideAny(x, y)) continue;
                nx[v] = x;
                ny[v] = y;
                v++;
            }
        }

        //A*: la visibilidad entre nodos se calcula solo cuando hace falta
        for(int i = 0; i < v; i++){
            g[i] = Float.MAX_VALUE;
            prev[i] = -1;
            closed[i] = false;
        }
        g[0] = 0f;

        while(true){
            int u = -1;
            float best = Float.MAX_VALUE;
            for(int i = 0; i < v; i++){
                if(closed[i] || g[i] == Float.MAX_VALUE) continue;
                float dx = nx[i] - tx, dy = ny[i] - ty;
                float f = g[i] + (float)Math.sqrt(dx * dx + dy * dy);
                if(f < best){
                    best = f;
                    u = i;
                }
            }
            if(u < 0) return false;
            if(u == 1) break;
            closed[u] = true;

            for(int w = 1; w < v; w++){
                if(closed[w] || w == u) continue;
                float dx = nx[w] - nx[u], dy = ny[w] - ny[u];
                float nd = g[u] + (float)Math.sqrt(dx * dx + dy * dy);
                if(nd >= g[w]) continue;
                if(!segmentFree(nx[u], ny[u], nx[w], ny[w], 1.03f, 3f)) continue;
                g[w] = nd;
                prev[w] = u;
            }
        }

        //reconstruir la ruta (sin el inicio)
        int len = 0;
        for(int u = 1; u != 0; u = prev[u]) len++;
        if(route.length < len * 2) route = new float[len * 2 * 2];
        int k = len;
        for(int u = 1; u != 0; u = prev[u]){
            k--;
            route[k * 2] = nx[u];
            route[k * 2 + 1] = ny[u];
        }
        routeLen = len;
        routeIdx = 0;
        routeVersion = version;
        routeTx = tx;
        routeTy = ty;
        return true;
    }

    private boolean insideAny(float x, float y){
        int n = size / 3;
        for(int i = 0; i < n; i++){
            float dx = x - data[i * 3], dy = y - data[i * 3 + 1], r = data[i * 3 + 2];
            if(dx * dx + dy * dy < r * r) return true;
        }
        return false;
    }

    /** Rodeo simple (como mucho 3 desvios encadenados alrededor de circulos): respaldo cuando no hay ruta planificada. */
    private boolean legacyWaypoint(float px, float py, float tx, float ty, float[] out){
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
