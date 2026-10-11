package safesupport;

import arc.*;
import arc.util.*;

import static mindustry.Vars.*;

/** Categoria "Safe Support AI" en Configuracion: deslizadores para los valores que antes solo se cambiaban recompilando. */
public class SafeSettings{
    private static boolean registered;

    public static void register(){
        if(registered || headless || ui == null || ui.settings == null) return;
        registered = true;

        title("ss-scan", tr("Radio de búsqueda de amenazas", "Threat search radius"),
            tr("Hasta dónde busca enemigos y torretas alrededor de cada unidad.", "How far each unit looks for enemies and turrets."));
        title("ss-unitpad", tr("Margen sobre el alcance de enemigos", "Margin over enemy range"),
            tr("Casillas que se suman al alcance de una unidad enemiga.", "Tiles added to an enemy unit's range."));
        title("ss-turretpad", tr("Margen sobre el alcance de torretas", "Margin over turret range"),
            tr("Casillas que se suman al alcance de una torreta enemiga.", "Tiles added to an enemy turret's range."));
        title("ss-minunit", tr("Peligro mínimo de un enemigo", "Minimum enemy danger radius"),
            tr("Radio mínimo de peligro de una unidad enemiga, aunque su alcance sea corto.", "Smallest danger radius of an enemy unit, even if its range is short."));
        title("ss-coversearch", tr("Búsqueda de muros para cubrirse", "Wall search for cover"),
            tr("Hasta dónde busca un muro propio para cubrirse (poly y mega).", "How far to look for an own wall to hide behind (poly and mega)."));
        title("ss-covergap", tr("Distancia detrás del muro", "Distance behind the wall"),
            tr("Distancia entre el borde del muro y la unidad. Las balas que explotan al chocar hacen daño en área.", "Distance between the wall edge and the unit. Bullets that explode on impact deal area damage."));
        title("ss-oresearch", tr("Búsqueda de veta alternativa (mono)", "Alternative ore search (mono)"),
            tr("Hasta dónde busca el mono otra veta segura, medido desde el núcleo.", "How far the mono looks for another safe ore, measured from the core."));
        title("ss-debug", tr("Mostrar zonas de peligro", "Show danger zones"),
            tr("Dibuja las zonas sobre el mapa (de las unidades seleccionadas). Naranja: un muro frena a la amenaza. Rojo: atraviesa muros. Anillo verde: unidad cubierta. Anillo rojo: expuesta. Punto y línea verdes: muro elegido.",
               "Draws the zones on the map (for selected units). Orange: a wall stops the threat. Red: it passes walls. Green ring: unit covered. Red ring: exposed. Green dot and line: chosen wall."));

        String tiles = tr(" casillas", " tiles");
        ui.settings.addCategory("Safe Support AI", t -> {
            t.sliderPref("ss-scan", 80, 20, 120, 5, v -> v + tiles);
            t.sliderPref("ss-unitpad", 4, 0, 10, 1, v -> v + tiles);
            t.sliderPref("ss-turretpad", 3, 0, 10, 1, v -> v + tiles);
            t.sliderPref("ss-minunit", 6, 2, 15, 1, v -> v + tiles);
            t.sliderPref("ss-coversearch", 10, 4, 20, 1, v -> v + tiles);
            t.sliderPref("ss-covergap", 15, 5, 40, 5, v -> Strings.fixed(v / 10f, 1) + tiles);
            t.sliderPref("ss-oresearch", 40, 10, 100, 5, v -> v + tiles);
            t.checkPref("ss-debug", false);
        });
    }

    private static void title(String name, String title, String description){
        Core.bundle.getProperties().put("setting." + name + ".name", title);
        Core.bundle.getProperties().put("setting." + name + ".description", description);
    }

    private static String tr(String es, String en){
        return "es".equals(Core.bundle.getLocale().getLanguage()) ? es : en;
    }
}
